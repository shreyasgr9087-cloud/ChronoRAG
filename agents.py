import os
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

raw_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = instructor.from_groq(raw_client, mode=instructor.Mode.JSON)

# --- 1. ROUTER SCHEMAS ---
class TemporalIntent(BaseModel):
    intent_type: str = Field(description="Must be 'current', 'historical', or 'timeline'")
    target_year: Optional[str] = Field(None, description="Extracted target year if mentioned, e.g., '2022'")
    reasoning: str = Field(description="Brief explanation of user intent")

# --- 2. RECONCILIATION SKEPTIC SCHEMAS ---
class ConflictDetail(BaseModel):
    invalidated_doc_id: str = Field(description="The ID of the superseded chunk")
    valid_doc_id: str = Field(description="The ID of the active/newer chunk")
    conflict_reason: str = Field(description="Why valid_doc_id supersedes invalidated_doc_id (e.g., newer effective date or table override)")

class ConflictReport(BaseModel):
    has_conflicts: bool = Field(description="True if factual contradictions exist across timestamps")
    conflicts: List[ConflictDetail] = Field(default_factory=list)
    surviving_doc_ids: List[str] = Field(description="List of doc IDs that are valid for generation")
    confidence_score: float = Field(description="Score between 0.0 and 1.0")

# --- AGENT FUNCTIONS ---
def run_temporal_router(user_query: str) -> TemporalIntent:
    """Classifies user intent to determine temporal constraints."""
    prompt = f"""Analyze the following query for temporal orientation.

Query: '{user_query}'

Rules:
- If the user is asking about the CURRENT or LATEST state of something (or asks without specifying a time), classify as 'current'.
- If the user is asking about a SPECIFIC past time period (e.g., "in 2022", "back in 2020"), classify as 'historical' and extract the target_year.
- If the user is asking about HOW something CHANGED over time, or comparing past vs present, classify as 'timeline'.
- For timeline queries that also mention a specific year, set target_year to that year."""
    
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=TemporalIntent,
        messages=[
            {"role": "system", "content": "You are a temporal query classification agent."},
            {"role": "user", "content": prompt}
        ]
    )

def run_reconciliation_skeptic(user_query: str, retrieved_chunks: List[dict], temporal_intent: TemporalIntent) -> ConflictReport:
    """Audits retrieved context for cross-modal and temporal conflicts.
    
    Behavior adapts based on temporal intent:
    - 'current': Aggressively invalidate outdated documents
    - 'historical': Do NOT invalidate — all docs survive, only flag conflicts for transparency
    - 'timeline': Do NOT invalidate — all docs survive, flag conflicts as evolution points
    """
    # Build the set of valid chunk IDs for post-validation
    valid_chunk_ids = {chunk["id"] for chunk in retrieved_chunks}
    
    formatted_context = ""
    for idx, chunk in enumerate(retrieved_chunks):
        formatted_context += f"\n--- CHUNK ID: {chunk['id']} ---\n"
        formatted_context += f"Effective Date: {chunk['metadata'].get('effective_date')}\n"
        formatted_context += f"Type: {chunk['metadata'].get('doc_type')}\n"
        formatted_context += f"Content:\n{chunk['document']}\n"

    # Adapt prompt based on temporal intent
    if temporal_intent.intent_type == "current":
        intent_instruction = """You are auditing for a CURRENT-STATE query. Your job is to aggressively identify outdated documents.
If a newer document supersedes an older one on the same topic, INVALIDATE the older document. 
Only the most current, authoritative versions should survive."""
    elif temporal_intent.intent_type == "historical":
        intent_instruction = f"""You are auditing for a HISTORICAL query targeting the year {temporal_intent.target_year or 'unknown'}.
DO NOT invalidate documents from the requested time period — the user specifically wants historical information.
Flag temporal conflicts for transparency, but ALL documents should survive in the surviving_doc_ids list.
The user needs to see what the policy/state was at that historical point in time."""
    else:  # timeline
        intent_instruction = """You are auditing for a TIMELINE/EVOLUTION query. The user wants to see how things changed over time.
DO NOT invalidate any documents — ALL documents should survive in the surviving_doc_ids list.
Flag temporal conflicts to show the evolution (which document supersedes which), but keep all versions
because the user needs the full chronological picture."""

    prompt = f"""User Query: '{user_query}'

Temporal Intent: {temporal_intent.intent_type} (target year: {temporal_intent.target_year or 'N/A'})

{intent_instruction}

Retrieved Context Chunks:
{formatted_context}

CRITICAL RULES:
1. You may ONLY reference chunk IDs from the list above. The valid IDs are: {list(valid_chunk_ids)}
2. Do NOT invent or hallucinate document IDs that are not in the retrieved chunks.
3. A document cannot invalidate itself — invalidated_doc_id and valid_doc_id must be different.
4. Only flag conflicts between documents that discuss the SAME topic.
5. The surviving_doc_ids list must be a subset of the valid IDs listed above.

Task: Audit these chunks according to the temporal intent rules above. Return a structured conflict report."""

    report = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=ConflictReport,
        messages=[
            {"role": "system", "content": "You are an aggressive legal and temporal skeptic auditor. Detect cross-modal and date conflicts. You must only reference document IDs that are explicitly provided to you."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Post-LLM validation: strip hallucinated doc IDs
    validated_conflicts = []
    for conflict in report.conflicts:
        if (conflict.invalidated_doc_id in valid_chunk_ids and 
            conflict.valid_doc_id in valid_chunk_ids and
            conflict.invalidated_doc_id != conflict.valid_doc_id):
            validated_conflicts.append(conflict)
    
    validated_surviving = [doc_id for doc_id in report.surviving_doc_ids if doc_id in valid_chunk_ids]
    
    # For historical/timeline: ensure all chunk IDs survive
    if temporal_intent.intent_type in ("historical", "timeline"):
        validated_surviving = list(valid_chunk_ids)
    
    # Safety: if validation stripped everything, keep all chunks
    if not validated_surviving:
        validated_surviving = list(valid_chunk_ids)
    
    report.conflicts = validated_conflicts
    report.surviving_doc_ids = validated_surviving
    report.has_conflicts = len(validated_conflicts) > 0
    
    return report

def run_synthesizer(user_query: str, valid_chunks: List[dict], conflict_report: ConflictReport, temporal_intent: TemporalIntent) -> str:
    """Generates final user-facing response backed by audited context.
    
    Adapts synthesis style based on temporal intent:
    - 'current': Authoritative answer citing latest docs, note what was superseded
    - 'historical': Present historical state as-is, note it may be outdated
    - 'timeline': Chronological narrative showing evolution
    """
    context_str = "\n\n".join([f"[{c['id']} - {c['metadata'].get('effective_date')}]: {c['document']}" for c in valid_chunks])
    
    # Adapt synthesis instructions based on intent
    if temporal_intent.intent_type == "current":
        synthesis_instruction = """Answer the user's query with the CURRENT/LATEST information only.
If older information was overridden by a newer table or policy, explicitly state what was superseded.
Cite the document IDs and effective dates of the sources you use."""
    elif temporal_intent.intent_type == "historical":
        target = temporal_intent.target_year or "the requested time period"
        synthesis_instruction = f"""The user is asking about HISTORICAL information from {target}.
Present the information AS IT WAS at that time, citing the relevant historical document directly.
You may note that the policy has since been updated, but the PRIMARY answer should describe the historical state.
Do NOT dismiss the historical document as 'no longer effective' — the user explicitly wants to know what it said."""
    else:  # timeline
        synthesis_instruction = """The user wants to understand HOW things CHANGED over time.
Present a CHRONOLOGICAL narrative showing the evolution from oldest to newest.
For each time period, cite the specific document and its effective date.
Highlight what changed between versions (e.g., amounts, rules, requirements).
Structure your answer as a clear before-and-after or timeline."""
    
    prompt = f"""User Query: {user_query}

Temporal Intent: {temporal_intent.intent_type}

Audited Context:
{context_str}

Conflict Audit Log:
{conflict_report.model_dump_json()}

Instructions: {synthesis_instruction}"""

    response = raw_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a professional corporate intelligence assistant. You provide precise, well-cited answers grounded exclusively in the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content