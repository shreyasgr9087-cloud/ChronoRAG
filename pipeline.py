import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from agents import run_temporal_router, run_reconciliation_skeptic, run_synthesizer

client = chromadb.PersistentClient(path="./chromadb_store")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection("chronorag_knowledge", embedding_function=default_ef)

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Fuses two ranked lists using the RRF algorithm.
    k=60 is the industry standard smoothing constant.
    """
    fused_scores = {}
    
    # Process Dense (Vector) Results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"doc": doc, "score": 0.0}
        fused_scores[doc_id]["score"] += 1.0 / (rank + k)
        
    # Process Sparse (BM25) Results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"doc": doc, "score": 0.0}
        fused_scores[doc_id]["score"] += 1.0 / (rank + k)
        
    # Sort by the fused score descending
    reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked]

def _boost_local_documents(fused_candidates, boost_factor=2.0):
    """Re-rank to prioritize local policy documents over TempLAMA benchmark noise.
    
    Local documents (source=local_file) are boosted by the given factor
    so they appear before irrelevant TempLAMA entries that may have
    incidental keyword overlap.
    """
    scored = []
    for rank, doc in enumerate(fused_candidates):
        base_score = 1.0 / (rank + 1)
        source = doc.get("metadata", {}).get("source", "")
        if source == "local_file":
            base_score *= boost_factor
        scored.append({"doc": doc, "score": base_score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in scored]

def execute_chronorag_pipeline(user_query: str):
    # 1. Temporal Router — classify query intent
    router_output = run_temporal_router(user_query)
    
    # 2. Extract All Documents for BM25 Indexing (In production, cache this)
    all_data = collection.get()
    all_docs = []
    for i in range(len(all_data["ids"])):
        all_docs.append({
            "id": all_data["ids"][i],
            "document": all_data["documents"][i],
            "metadata": all_data["metadatas"][i]
        })
    
    # 3. Sparse Retrieval (BM25 Keyword Search)
    tokenized_corpus = [doc["document"].lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = user_query.lower().split()
    
    # Get top 10 BM25 matches
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    sparse_results = [all_docs[i] for i in top_bm25_indices]

    # 4. Dense Retrieval (ChromaDB Semantic Search)
    vector_res = collection.query(query_texts=[user_query], n_results=10)
    dense_results = []
    if vector_res["documents"]:
        for i in range(len(vector_res["documents"][0])):
            dense_results.append({
                "id": vector_res["ids"][0][i],
                "document": vector_res["documents"][0][i],
                "metadata": vector_res["metadatas"][0][i]
            })

    # 5. Reciprocal Rank Fusion
    fused_candidates = reciprocal_rank_fusion(dense_results, sparse_results)
    
    # 5b. Boost local documents over TempLAMA noise (B6 fix)
    fused_candidates = _boost_local_documents(fused_candidates)
    
    # Take only the top 5 absolute best chunks to pass to the LLM
    final_retrieved_chunks = fused_candidates[:5]

    # 6. Reconciliation Skeptic Audit — now receives temporal intent (B2 fix)
    conflict_report = run_reconciliation_skeptic(user_query, final_retrieved_chunks, router_output)

    # 7. Filter and Synthesize
    surviving_ids = set(conflict_report.surviving_doc_ids)
    valid_chunks = [c for c in final_retrieved_chunks if c["id"] in surviving_ids]
    
    if not valid_chunks:
        valid_chunks = final_retrieved_chunks 

    # 8. Synthesizer now receives temporal intent for context-aware generation (B2 fix)
    final_answer = run_synthesizer(user_query, valid_chunks, conflict_report, router_output)

    return {
        "answer": final_answer,
        "router": router_output.model_dump(),
        "conflict_report": conflict_report.model_dump(),
        "retrieved_chunks": final_retrieved_chunks
    }