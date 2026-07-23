import os
import glob
import re
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

DB_PATH = "./chromadb_store"
LOCAL_DOCS_DIR = "./rag_docs"

client = chromadb.PersistentClient(path=DB_PATH)
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="chronorag_knowledge",
    embedding_function=default_ef
)

# ──────────────────────────────────────────────────────────────
# Frontmatter Parsing
# ──────────────────────────────────────────────────────────────

# Canonical key mappings for frontmatter normalization
_KEY_ALIASES = {
    "date": "effective_date",
    "published": "effective_date",
    "updated": "effective_date",
    "created": "effective_date",
    "type": "doc_type",
}

def parse_frontmatter(file_content):
    """Extracts metadata from markdown frontmatter (--- key: val ---).
    
    Normalizes common key aliases (e.g., 'date' -> 'effective_date')
    so the pipeline always has a consistent metadata schema.
    """
    metadata = {
        "effective_date": "2026-01-01",
        "doc_type": "unstructured_text",
        "title": "Untitled Document"
    }
    
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", file_content, re.DOTALL)
    if match:
        yaml_text, body = match.groups()
        for line in yaml_text.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Normalize aliases to canonical keys
                canonical_key = _KEY_ALIASES.get(k, k)
                metadata[canonical_key] = v
        return metadata, body
    return metadata, file_content


# ──────────────────────────────────────────────────────────────
# Markdown-Aware Chunking
# ──────────────────────────────────────────────────────────────

# Chunking thresholds (in characters)
_CHUNK_SIZE = 1000          # Target size for text sub-splits
_CHUNK_OVERLAP = 200        # Overlap between consecutive text chunks
_MIN_DOC_SIZE = 800         # Documents below this stay as a single chunk
_MAX_TABLE_CHUNK = 3000     # Tables under this size are kept atomic

# Header hierarchy for structural splitting
_HEADERS_TO_SPLIT_ON = [
    ("#",   "header_1"),
    ("##",  "header_2"),
    ("###", "header_3"),
]


def _contains_markdown_table(text: str) -> bool:
    """Detects Markdown tables by looking for pipe-delimited rows.
    
    Requires at least 3 pipe-rows (header + separator + ≥1 data row)
    to distinguish real tables from stray pipe characters.
    """
    lines = text.strip().split("\n")
    pipe_rows = [ln for ln in lines if ln.strip().startswith("|") and ln.strip().endswith("|")]
    return len(pipe_rows) >= 3


def chunk_markdown_document(body: str, metadata: dict, base_doc_id: str):
    """Splits a Markdown document into chunks while preserving structure.
    
    Strategy:
    1. Small documents (< _MIN_DOC_SIZE chars) → single chunk.
    2. Larger documents → split on Markdown headers (##, ###).
    3. Table-containing sections are kept ATOMIC (never split mid-table).
    4. Long text-only sections are sub-split with RecursiveCharacterTextSplitter.
    5. Every child chunk inherits the parent document's metadata.
    
    Returns:
        tuple: (texts, metadatas, ids) ready for ChromaDB insertion.
    """
    body_stripped = body.strip()
    
    # Small documents: keep as single chunk
    if len(body_stripped) <= _MIN_DOC_SIZE:
        chunk_meta = {**metadata}
        if _contains_markdown_table(body_stripped):
            chunk_meta["doc_type"] = "structured_table"
        return [body_stripped], [chunk_meta], [f"{base_doc_id}-chunk-0"]
    
    # Step 1: Split on Markdown headers (keeps headers in content)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_splits = md_splitter.split_text(body)
    
    # Step 2: For each section, decide: keep atomic or sub-split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    
    texts, metadatas, ids = [], [], []
    chunk_idx = 0
    
    for split in header_splits:
        content = split.page_content.strip()
        if not content:
            continue
        
        has_table = _contains_markdown_table(content)
        
        if has_table and len(content) <= _MAX_TABLE_CHUNK:
            # Table section: keep atomic, tag as structured_table
            chunk_meta = {**metadata, "doc_type": "structured_table"}
            texts.append(content)
            metadatas.append(chunk_meta)
            ids.append(f"{base_doc_id}-chunk-{chunk_idx}")
            chunk_idx += 1
        elif len(content) > _CHUNK_SIZE:
            # Long text section: sub-split with overlap
            sub_chunks = text_splitter.split_text(content)
            for sub in sub_chunks:
                chunk_meta = {**metadata}
                if _contains_markdown_table(sub):
                    chunk_meta["doc_type"] = "structured_table"
                texts.append(sub)
                metadatas.append(chunk_meta)
                ids.append(f"{base_doc_id}-chunk-{chunk_idx}")
                chunk_idx += 1
        else:
            # Normal-sized section: keep as-is
            chunk_meta = {**metadata}
            texts.append(content)
            metadatas.append(chunk_meta)
            ids.append(f"{base_doc_id}-chunk-{chunk_idx}")
            chunk_idx += 1
    
    # Fallback: if no chunks produced, keep entire body as one chunk
    if not texts:
        chunk_meta = {**metadata}
        if _contains_markdown_table(body_stripped):
            chunk_meta["doc_type"] = "structured_table"
        texts.append(body_stripped)
        metadatas.append(chunk_meta)
        ids.append(f"{base_doc_id}-chunk-0")
    
    return texts, metadatas, ids


# ──────────────────────────────────────────────────────────────
# Document Sync & Ingestion
# ──────────────────────────────────────────────────────────────

def sync_local_documents():
    """Scans ./rag_docs, chunks each file, and syncs into ChromaDB.
    
    Uses delete-then-add (not upsert) to prevent stale metadata keys
    from persisting across schema changes.
    """
    if not os.path.exists(LOCAL_DOCS_DIR):
        os.makedirs(LOCAL_DOCS_DIR)
        print(f"Created '{LOCAL_DOCS_DIR}'. Add custom .md or .txt files here.")
        return 0

    files = glob.glob(os.path.join(LOCAL_DOCS_DIR, "*.*"))
    if not files:
        return 0

    all_texts, all_metadatas, all_ids = [], [], []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, body = parse_frontmatter(content)
        metadata["source"] = "local_file"
        base_doc_id = f"LOCAL-{os.path.basename(filepath)}"

        # Chunk the document with structure-aware splitting
        texts, metadatas, ids = chunk_markdown_document(body, metadata, base_doc_id)
        all_texts.extend(texts)
        all_metadatas.extend(metadatas)
        all_ids.extend(ids)

    # Clean slate: delete ALL existing local chunks by metadata filter,
    # then add fresh. This prevents orphan chunks from prior ingestions
    # that may have produced a different number of chunks per document.
    existing = collection.get(where={"source": "local_file"})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    
    collection.add(documents=all_texts, metadatas=all_metadatas, ids=all_ids)
    print(f"Synced {len(all_ids)} chunks from {len(files)} document(s).")
    return len(all_ids)


def ingest_templama_benchmark(limit=500):
    """Ingests TempLAMA with cross-modal format transformation."""
    print("Ingesting TempLAMA benchmark dataset...")
    dataset = load_dataset("Yova/templama", split=f"train[:{limit}]")

    texts, metadatas, ids = [], [], []
    for i, row in enumerate(dataset):
        query, answer, date = row['query'], row['answer'], str(row['date'])
        
        if int(date) < 2010:
            doc_type = "unstructured_text"
            doc_content = f"Historical Policy Note ({date}): Regarding '{query}', official stance is {answer}."
        else:
            doc_type = "structured_table"
            doc_content = f"# Active Schedule Table ({date})\n| Query Focus | Verified Decision | Status |\n| {query} | **{answer}** | Effective |"

        texts.append(doc_content)
        metadatas.append({
            "effective_date": f"{date}-01-01",
            "doc_type": doc_type,
            "title": f"TempLAMA-{query[:20]}",
            "source": "templama"
        })
        ids.append(f"TLAMA-{i}")

    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(ids)} benchmark records.")


if __name__ == "__main__":
    sync_local_documents()
    if collection.count() < 100:
        ingest_templama_benchmark(limit=500)
    print(f"Total documents in database: {collection.count()}")