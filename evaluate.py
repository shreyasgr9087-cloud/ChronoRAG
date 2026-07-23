"""
evaluate.py - ChronoRAG vs. Naive RAG  ·  Automated Evaluation Framework
=========================================================================
Runs 10 expert-crafted temporal queries through both a Naive RAG baseline
and the full ChronoRAG pipeline, then scores each answer with an LLM judge
on three axes:

    1. Temporal Accuracy   - Does the answer reflect the correct time period?
    2. Contradiction Handling - Does it flag/resolve conflicting documents?
    3. Citation Quality     - Are source dates/IDs referenced properly?

Outputs (written to ./eval_results/):
    • scorecard.csv          - raw per-query scores
    • summary.txt            - aggregate statistics
    • radar_comparison.png   - radar chart overlay
    • bar_comparison.png     - grouped bar chart
    • per_query_breakdown.png - per-query side-by-side breakdown
    • full_log.json          - every answer + judge reasoning (for auditing)

Usage:
    python evaluate.py
"""

import os
import sys
import json
import csv
import time
import math
import textwrap
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load .env FIRST, before any module that creates a Groq client
load_dotenv(override=True)

from groq import Groq
import instructor
from pydantic import BaseModel, Field
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define common retry wrapper for Groq API calls
groq_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError)),
    before_sleep=lambda retry_state: logger.warning(
        f"API call failed on attempt {retry_state.attempt_number}: {retry_state.outcome.exception()}. Retrying in {retry_state.next_action.sleep}s..."
    )
)

# -- Project imports ---------------------------------------------
from pipeline import execute_chronorag_pipeline

# -- Clients -----------------------------------------------------
raw_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
judge_client = instructor.from_groq(raw_client, mode=instructor.Mode.JSON)

chroma_client = chromadb.PersistentClient(path="./chromadb_store")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    "chronorag_knowledge", embedding_function=default_ef
)

# -- Output directory --------------------------------------------
RESULTS_DIR = "./eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================
#  EVALUATION QUERIES
# =============================================================
# Each query is designed to exploit a specific temporal failure mode
# that Naive RAG is prone to. The "expected_behavior" is the gold
# reference the LLM judge uses when grading.

EVAL_QUERIES: List[Dict[str, str]] = [
    # -- Q1: Current-state with superseded policy --
    {
        "id": "Q01",
        "query": "What is the current daily meal allowance for business travel?",
        "category": "current_state",
        "expected_behavior": (
            "Must answer $100/day citing the 2026 travel policy. "
            "Must NOT present the outdated $50/day figure from the 2022 memo "
            "as current. Ideally notes the 2022 policy was superseded."
        ),
    },
    # -- Q2: Historical query - should NOT invalidate old docs --
    {
        "id": "Q02",
        "query": "What was the remote work policy back in 2022?",
        "category": "historical",
        "expected_behavior": (
            "Must describe the 2022 policy: 2 remote days/week, no equipment "
            "stipend, core hours 9-5, VPN required. Must NOT dismiss the 2022 "
            "document as outdated - the user explicitly asked for historical info."
        ),
    },
    # -- Q3: Timeline / evolution query --
    {
        "id": "Q03",
        "query": "How have the company's salary bands for engineers changed over time?",
        "category": "timeline",
        "expected_behavior": (
            "Must present a chronological comparison: 2023 bands ($85k entry, "
            "$110-130k mid, $140-170k senior, 10% bonus, stock options) vs. "
            "2026 bands ($105-120k entry, $130-155k mid, $160-200k senior, "
            "12-25% bonus, RSUs). Should note key changes like RSU replacement "
            "and vesting schedule update."
        ),
    },
    # -- Q4: Contradiction detection - data retention --
    {
        "id": "Q04",
        "query": "Is cloud storage permitted for customer PII?",
        "category": "current_state",
        "expected_behavior": (
            "Must answer YES, citing the 2025 GDPR-aligned policy (cloud with "
            "AES-256 encryption). Must flag the contradiction with the 2021 "
            "policy that prohibited cloud storage. Should note the 2025 policy "
            "explicitly supersedes 2021."
        ),
    },
    # -- Q5: Status reversal - hiring freeze --
    {
        "id": "Q05",
        "query": "Is the company currently under a hiring freeze?",
        "category": "current_state",
        "expected_behavior": (
            "Must answer NO - the freeze was lifted in Feb 2024. Must NOT say "
            "the company is frozen (the 2023 freeze doc is outdated). Should "
            "cite both the 2023 freeze and the 2024 lift to show awareness."
        ),
    },
    # -- Q6: Specific numeric lookup - current --
    {
        "id": "Q06",
        "query": "What is the current base salary range for a Senior Engineer?",
        "category": "current_state",
        "expected_behavior": (
            "Must answer $160,000-$200,000 from the 2026 salary bands. "
            "Must NOT quote the 2023 figure of $140,000-$170,000 as current. "
            "Should cite the 2026 revision document."
        ),
    },
    # -- Q7: Policy evolution with structural change --
    {
        "id": "Q07",
        "query": "How has the data retention policy for customer PII evolved?",
        "category": "timeline",
        "expected_behavior": (
            "Must trace evolution: 2021 (indefinite retention, on-prem only, "
            "encryption optional, 45-day access requests) → 2025 (3-year max, "
            "cloud permitted, AES-256 mandatory, 30-day access requests). "
            "Should present both versions chronologically."
        ),
    },
    # -- Q8: Historical numeric detail --
    {
        "id": "Q08",
        "query": "What was the lodging reimbursement limit in 2022?",
        "category": "historical",
        "expected_behavior": (
            "Must answer $200 per night, citing the 2022 travel memo. "
            "Should NOT substitute the 2026 figure ($250/$350). May optionally "
            "note that the limit has since been updated."
        ),
    },
    # -- Q9: Cross-domain temporal reasoning --
    {
        "id": "Q09",
        "query": "What equipment stipend do Senior engineers get for remote work?",
        "category": "current_state",
        "expected_behavior": (
            "Must answer $1,000/year from the 2025 remote work policy (Senior "
            "tier = 2-5 years). Must note that no stipend existed under the "
            "2022 policy. Should cite the correct tier from the table."
        ),
    },
    # -- Q10: Ambiguous recency - performance reviews --
    {
        "id": "Q10",
        "query": "How does the company handle consecutive low performance ratings?",
        "category": "current_state",
        "expected_behavior": (
            "Must describe the 2024 performance review process: employees rated "
            "'Below Expectations' for two consecutive years are placed on a "
            "90-day PIP. Should cite the 2024 performance review document. "
            "There is only one version of this policy, so no conflict exists."
        ),
    },
]


# =============================================================
#  NAIVE RAG BASELINE
# =============================================================
# Deliberately simple: vector search → top-5 chunks → LLM.
# No temporal routing, no conflict detection, no date awareness.

@groq_retry
def execute_naive_rag(user_query: str) -> Dict[str, Any]:
    """
    Bare-bones RAG: embed query → cosine-similarity top-5 → synthesize.
    This is the 'control group' that ignores all temporal signals.
    """
    # 1. Vector retrieval only (no BM25, no RRF, no boosting)
    results = collection.query(query_texts=[user_query], n_results=5)

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            })

    # 2. Dump all chunks into a flat context string (no date annotations)
    context = "\n\n".join(
        [f"[{c['id']}]: {c['document']}" for c in chunks]
    )

    # 3. Simple synthesis - no temporal instructions
    prompt = f"""Answer the user's question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {user_query}"""

    response = raw_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful corporate assistant. Answer the "
                    "question using only the provided context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "retrieved_chunks": chunks,
    }


# =============================================================
#  LLM JUDGE
# =============================================================

class JudgeScore(BaseModel):
    temporal_accuracy: int = Field(
        description="1-5 score: does the answer reflect the correct time period?"
    )
    contradiction_handling: int = Field(
        description="1-5 score: does the answer identify and resolve contradictions?"
    )
    citation_quality: int = Field(
        description="1-5 score: are sources, dates, and document IDs cited?"
    )
    reasoning: str = Field(
        description="Brief explanation of the scores assigned"
    )


@groq_retry
def judge_answer(
    query: str,
    answer: str,
    expected_behavior: str,
    category: str,
) -> JudgeScore:
    """Ask an LLM to grade an answer against the gold-standard expectation."""

    prompt = f"""You are an expert evaluator for a Retrieval-Augmented Generation system
that handles temporal (time-sensitive) corporate documents.

QUERY: {query}
QUERY CATEGORY: {category}

EXPECTED BEHAVIOR (Gold Standard):
{expected_behavior}

ACTUAL ANSWER TO EVALUATE:
{answer}

Score the answer on three dimensions (1 = worst, 5 = best):

1. **Temporal Accuracy** (1-5):
   - 5: Perfectly reflects the correct time period; no outdated facts presented as current
   - 3: Partially correct but mixes time periods or is vague about dates
   - 1: Uses the wrong time period entirely (e.g., cites 2022 data for a "current" query)

2. **Contradiction Handling** (1-5):
   - 5: Explicitly identifies superseded documents and explains the conflict resolution
   - 3: Mentions multiple sources but doesn't clearly resolve which is authoritative
   - 1: Blindly merges contradictory documents without noticing the conflict

3. **Citation Quality** (1-5):
   - 5: Cites specific document names, effective dates, and/or chunk IDs
   - 3: Vaguely references "according to policy" without specifics
   - 1: No citations or source attribution at all

Provide your scores and a brief reasoning."""

    return judge_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=JudgeScore,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, impartial evaluation judge. "
                    "Score based ONLY on the criteria provided. "
                    "Do not give generous scores - be precise and critical."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )


# =============================================================
#  PLOTTING UTILITIES
# =============================================================

def _try_import_matplotlib():
    """Import matplotlib lazily; return None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        return None, None


def plot_radar_chart(naive_avgs: Dict[str, float], chrono_avgs: Dict[str, float], path: str):
    """Generate a radar (spider) chart comparing average scores."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        print("  [!] matplotlib not installed - skipping radar chart")
        return

    categories = list(naive_avgs.keys())
    N = len(categories)

    # Compute angles for each axis
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    naive_vals = [naive_avgs[c] for c in categories] + [naive_avgs[categories[0]]]
    chrono_vals = [chrono_avgs[c] for c in categories] + [chrono_avgs[categories[0]]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Style
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.plot(angles, naive_vals, "o-", linewidth=2.5, color="#ff6b6b", label="Naive RAG", markersize=8)
    ax.fill(angles, naive_vals, alpha=0.15, color="#ff6b6b")

    ax.plot(angles, chrono_vals, "o-", linewidth=2.5, color="#51cf66", label="ChronoRAG", markersize=8)
    ax.fill(angles, chrono_vals, alpha=0.15, color="#51cf66")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [c.replace("_", "\n").title() for c in categories],
        size=12, color="white", fontweight="bold",
    )
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=9, color="#888")
    ax.tick_params(colors="#888")

    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color="#333", linestyle="--", linewidth=0.5)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.25, 1.1),
        fontsize=12, facecolor="#161b22", edgecolor="#333",
        labelcolor="white",
    )
    ax.set_title(
        "ChronoRAG vs Naive RAG - Average Scores",
        size=16, color="white", fontweight="bold", pad=30,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [OK] Saved radar chart → {path}")


def plot_bar_chart(naive_avgs: Dict[str, float], chrono_avgs: Dict[str, float], path: str):
    """Generate a grouped bar chart comparing average scores."""
    plt, ticker = _try_import_matplotlib()
    if plt is None:
        print("  [!] matplotlib not installed - skipping bar chart")
        return

    import numpy as np

    categories = list(naive_avgs.keys())
    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    naive_vals = [naive_avgs[c] for c in categories]
    chrono_vals = [chrono_avgs[c] for c in categories]

    bars1 = ax.bar(x - width / 2, naive_vals, width, label="Naive RAG",
                   color="#ff6b6b", edgecolor="#ff4444", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width / 2, chrono_vals, width, label="ChronoRAG",
                   color="#51cf66", edgecolor="#37b24d", linewidth=0.8, zorder=3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.1f}",
                ha="center", va="bottom", fontsize=11, color="#ff6b6b", fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.1f}",
                ha="center", va="bottom", fontsize=11, color="#51cf66", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in categories],
        fontsize=12, color="white", fontweight="bold",
    )
    ax.set_ylabel("Score (1-5)", fontsize=13, color="white", fontweight="bold")
    ax.set_ylim(0, 5.8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="y", colors="#888")

    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.5, zorder=0)
    ax.legend(fontsize=12, facecolor="#161b22", edgecolor="#333", labelcolor="white")
    ax.set_title(
        "ChronoRAG vs Naive RAG - Score Comparison",
        fontsize=16, color="white", fontweight="bold", pad=15,
    )

    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [OK] Saved bar chart → {path}")


def plot_per_query_breakdown(results: List[Dict], path: str):
    """Generate a per-query grouped bar chart showing all 3 dimensions."""
    plt, ticker = _try_import_matplotlib()
    if plt is None:
        print("  [!] matplotlib not installed - skipping per-query chart")
        return

    import numpy as np

    n = len(results)
    x = np.arange(n)
    width = 0.13
    dims = ["temporal_accuracy", "contradiction_handling", "citation_quality"]
    dim_labels = ["Temporal", "Contradiction", "Citation"]
    naive_colors = ["#ff8787", "#ffa8a8", "#ffc9c9"]
    chrono_colors = ["#69db7c", "#8ce99a", "#b2f2bb"]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, dim in enumerate(dims):
        naive_vals = [r["naive_scores"][dim] for r in results]
        chrono_vals = [r["chrono_scores"][dim] for r in results]

        ax.bar(x - width * 3 + width * i, naive_vals, width,
               color=naive_colors[i], edgecolor="#ff4444", linewidth=0.4,
               label=f"Naive {dim_labels[i]}" if True else "", zorder=3)
        ax.bar(x + width * i, chrono_vals, width,
               color=chrono_colors[i], edgecolor="#37b24d", linewidth=0.4,
               label=f"Chrono {dim_labels[i]}" if True else "", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["query_id"] for r in results],
        fontsize=11, color="white", fontweight="bold",
    )
    ax.set_ylabel("Score (1-5)", fontsize=13, color="white", fontweight="bold")
    ax.set_ylim(0, 5.8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="y", colors="#888")

    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.5, zorder=0)
    ax.legend(
        fontsize=9, facecolor="#161b22", edgecolor="#333",
        labelcolor="white", ncol=6, loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    ax.set_title(
        "Per-Query Score Breakdown (All Dimensions)",
        fontsize=16, color="white", fontweight="bold", pad=40,
    )

    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [OK] Saved per-query breakdown → {path}")


# =============================================================
#  MAIN EVALUATION LOOP
# =============================================================

def run_evaluation():
    """Execute the full evaluation suite and save all outputs."""
    print("=" * 70)
    print("  ChronoRAG vs Naive RAG  -  Automated Evaluation Framework")
    print("=" * 70)
    print(f"  Queries:  {len(EVAL_QUERIES)}")
    print(f"  Output:   {os.path.abspath(RESULTS_DIR)}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    full_log = []

    for idx, q in enumerate(EVAL_QUERIES):
        qid = q["id"]
        query_text = q["query"]
        category = q["category"]
        expected = q["expected_behavior"]

        print(f"\n{'-' * 60}")
        print(f"  [{qid}] ({category.upper()}) {query_text}")
        print(f"{'-' * 60}")

        # -- Run Naive RAG ---------------------------------------
        print("  > Running Naive RAG ...", end=" ", flush=True)
        t0 = time.time()
        naive_result = {}
        try:
            naive_result = execute_naive_rag(query_text)
            naive_answer = naive_result["answer"]
            naive_time = time.time() - t0
            print(f"done ({naive_time:.1f}s)")
        except Exception as e:
            naive_answer = f"[ERROR] {e}"
            naive_time = time.time() - t0
            print(f"FAILED ({e})")

        # -- Run ChronoRAG --------------------------------------
        print("  > Running ChronoRAG  ...", end=" ", flush=True)
        t0 = time.time()
        chrono_result = {}
        try:
            chrono_result = execute_chronorag_pipeline(query_text)
            chrono_answer = chrono_result["answer"]
            chrono_time = time.time() - t0
            print(f"done ({chrono_time:.1f}s)")
        except Exception as e:
            chrono_answer = f"[ERROR] {e}"
            chrono_time = time.time() - t0
            print(f"FAILED ({e})")

        # -- Judge both answers ---------------------------------
        print("  > Judging Naive RAG  ...", end=" ", flush=True)
        # Small delay to avoid rate limiting (Groq free-tier is ~30 req/min)
        time.sleep(2)
        try:
            naive_score = judge_answer(query_text, naive_answer, expected, category)
            print(f"T={naive_score.temporal_accuracy} C={naive_score.contradiction_handling} Q={naive_score.citation_quality}")
        except Exception as e:
            print(f"JUDGE ERROR: {e}")
            naive_score = JudgeScore(
                temporal_accuracy=1, contradiction_handling=1,
                citation_quality=1, reasoning=f"Judge failed: {e}",
            )

        print("  > Judging ChronoRAG  ...", end=" ", flush=True)
        time.sleep(2)
        try:
            chrono_score = judge_answer(query_text, chrono_answer, expected, category)
            print(f"T={chrono_score.temporal_accuracy} C={chrono_score.contradiction_handling} Q={chrono_score.citation_quality}")
        except Exception as e:
            print(f"JUDGE ERROR: {e}")
            chrono_score = JudgeScore(
                temporal_accuracy=1, contradiction_handling=1,
                citation_quality=1, reasoning=f"Judge failed: {e}",
            )

        # -- Collect results ------------------------------------
        row = {
            "query_id": qid,
            "category": category,
            "query": query_text,
            "naive_scores": {
                "temporal_accuracy": naive_score.temporal_accuracy,
                "contradiction_handling": naive_score.contradiction_handling,
                "citation_quality": naive_score.citation_quality,
            },
            "chrono_scores": {
                "temporal_accuracy": chrono_score.temporal_accuracy,
                "contradiction_handling": chrono_score.contradiction_handling,
                "citation_quality": chrono_score.citation_quality,
            },
            "naive_total": (
                naive_score.temporal_accuracy
                + naive_score.contradiction_handling
                + naive_score.citation_quality
            ),
            "chrono_total": (
                chrono_score.temporal_accuracy
                + chrono_score.contradiction_handling
                + chrono_score.citation_quality
            ),
            "naive_time": round(naive_time, 2),
            "chrono_time": round(chrono_time, 2),
        }
        results.append(row)

        # Full log entry (for auditing)
        log_entry = {
            **row,
            "expected_behavior": expected,
            "naive_answer": naive_answer,
            "chrono_answer": chrono_answer,
            "naive_judge_reasoning": naive_score.reasoning,
            "chrono_judge_reasoning": chrono_score.reasoning,
            "naive_chunks": [
                {"id": c["id"], "metadata": c["metadata"]}
                for c in (naive_result.get("retrieved_chunks", []) if isinstance(naive_result, dict) else [])
            ],
            "chrono_chunks": [
                {"id": c["id"], "metadata": c["metadata"]}
                for c in (chrono_result.get("retrieved_chunks", []) if isinstance(chrono_result, dict) else [])
            ],
        }
        full_log.append(log_entry)

        # Rate-limiting pause between queries
        if idx < len(EVAL_QUERIES) - 1:
            time.sleep(3)

    # -- Aggregate Statistics ------------------------------------
    dims = ["temporal_accuracy", "contradiction_handling", "citation_quality"]
    naive_avgs = {}
    chrono_avgs = {}
    for dim in dims:
        naive_avgs[dim] = round(
            sum(r["naive_scores"][dim] for r in results) / len(results), 2
        )
        chrono_avgs[dim] = round(
            sum(r["chrono_scores"][dim] for r in results) / len(results), 2
        )

    naive_total_avg = round(sum(r["naive_total"] for r in results) / len(results), 2)
    chrono_total_avg = round(sum(r["chrono_total"] for r in results) / len(results), 2)
    naive_time_avg = round(sum(r["naive_time"] for r in results) / len(results), 2)
    chrono_time_avg = round(sum(r["chrono_time"] for r in results) / len(results), 2)

    # Win/tie/loss per query
    chrono_wins = sum(1 for r in results if r["chrono_total"] > r["naive_total"])
    ties = sum(1 for r in results if r["chrono_total"] == r["naive_total"])
    naive_wins = sum(1 for r in results if r["chrono_total"] < r["naive_total"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SAVE OUTPUTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1. CSV Scorecard
    csv_path = os.path.join(RESULTS_DIR, "scorecard.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query ID", "Category", "Query",
            "Naive Temporal", "Naive Contradiction", "Naive Citation", "Naive Total",
            "Chrono Temporal", "Chrono Contradiction", "Chrono Citation", "Chrono Total",
            "Naive Time (s)", "Chrono Time (s)",
        ])
        for r in results:
            writer.writerow([
                r["query_id"], r["category"], r["query"],
                r["naive_scores"]["temporal_accuracy"],
                r["naive_scores"]["contradiction_handling"],
                r["naive_scores"]["citation_quality"],
                r["naive_total"],
                r["chrono_scores"]["temporal_accuracy"],
                r["chrono_scores"]["contradiction_handling"],
                r["chrono_scores"]["citation_quality"],
                r["chrono_total"],
                r["naive_time"], r["chrono_time"],
            ])
    print(f"\n  [OK] Saved scorecard → {csv_path}")

    # 2. Full JSON log
    log_path = os.path.join(RESULTS_DIR, "full_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_log, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved full log → {log_path}")

    # 3. Summary text
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    improvement = (
        round(((chrono_total_avg - naive_total_avg) / naive_total_avg) * 100, 1)
        if naive_total_avg > 0 else 0
    )

    summary = textwrap.dedent(f"""\
    ╔══════════════════════════════════════════════════════════════╗
    ║       ChronoRAG vs Naive RAG - Evaluation Summary          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<52} ║
    ║  Queries: {len(results):<49} ║
    ║  Model: llama-3.3-70b-versatile (Groq)                      ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  DIMENSION AVERAGES (1-5 scale)                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                        Naive RAG    ChronoRAG    Delta       ║
    ║  Temporal Accuracy     {naive_avgs['temporal_accuracy']:<13}{chrono_avgs['temporal_accuracy']:<13}+{chrono_avgs['temporal_accuracy'] - naive_avgs['temporal_accuracy']:.2f}        ║
    ║  Contradiction Handling{naive_avgs['contradiction_handling']:<13}{chrono_avgs['contradiction_handling']:<13}+{chrono_avgs['contradiction_handling'] - naive_avgs['contradiction_handling']:.2f}        ║
    ║  Citation Quality      {naive_avgs['citation_quality']:<13}{chrono_avgs['citation_quality']:<13}+{chrono_avgs['citation_quality'] - naive_avgs['citation_quality']:.2f}        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  TOTALS (out of 15)                                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Naive RAG avg total:    {naive_total_avg:<10}                       ║
    ║  ChronoRAG avg total:    {chrono_total_avg:<10}                       ║
    ║  Overall improvement:    {improvement:+.1f}%                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  HEAD-TO-HEAD                                                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  ChronoRAG wins: {chrono_wins:<5}  Ties: {ties:<5}  Naive wins: {naive_wins:<5}      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  LATENCY                                                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Naive RAG avg:    {naive_time_avg:.2f}s                                 ║
    ║  ChronoRAG avg:    {chrono_time_avg:.2f}s                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  [OK] Saved summary → {summary_path}")

    # 4. Charts
    print("\n  Generating charts ...")
    plot_radar_chart(
        naive_avgs, chrono_avgs,
        os.path.join(RESULTS_DIR, "radar_comparison.png"),
    )
    plot_bar_chart(
        naive_avgs, chrono_avgs,
        os.path.join(RESULTS_DIR, "bar_comparison.png"),
    )
    plot_per_query_breakdown(
        results,
        os.path.join(RESULTS_DIR, "per_query_breakdown.png"),
    )

    # -- Final Console Summary ----------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Naive RAG avg:    {naive_total_avg}/15")
    print(f"  ChronoRAG avg:    {chrono_total_avg}/15")
    print(f"  Improvement:      {improvement:+.1f}%")
    print(f"  Wins/Ties/Losses: {chrono_wins}/{ties}/{naive_wins}")
    print(f"  Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_evaluation()
