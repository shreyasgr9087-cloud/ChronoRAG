"""
ChronoRAG Pipeline Test Suite
=============================
Runs 10 queries against the pipeline, collects all agent outputs,
and writes a structured JSON report for analysis.
"""

import json
import time
import traceback
from datetime import datetime

# Re-ingest local documents before testing
from ingest import sync_local_documents
print("=" * 60)
print("STEP 1: Re-syncing local documents into ChromaDB...")
count = sync_local_documents()
print(f"Synced {count} documents.")
print("=" * 60)

from pipeline import execute_chronorag_pipeline

# ──────────────────────────────────────────────────────────────
# TEST QUERIES — 10 queries covering all pipeline capabilities
# ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "id": "Q01",
        "query": "What is the current remote work policy?",
        "category": "CURRENT_STATE",
        "expected_behavior": "Should return the 2025 policy (3-5 remote days by seniority, equipment stipend, flexible hours). Must NOT blend the 2022 policy (2 days, no stipend, 9-5 hours). The Skeptic should flag the 2022 doc as superseded.",
        "expected_facts": [
            "Junior employees get 3 remote days per week",
            "Equipment stipend is $500-$1500/year",
            "Core hours are flexible 10 AM - 4 PM overlap",
            "2022 policy (2 days, no stipend) is explicitly superseded"
        ]
    },
    {
        "id": "Q02",
        "query": "What was the remote work policy in 2022?",
        "category": "HISTORICAL_SNAPSHOT",
        "expected_behavior": "Should return the 2022 policy (2 days/week, no equipment stipend, 9-5 core hours). Router should classify as 'historical' with target_year='2022'. May also mention the 2025 update as context but should primarily describe 2022 state.",
        "expected_facts": [
            "2 days per week remote",
            "No equipment stipend",
            "Core hours 9 AM - 5 PM",
            "VPN access mandatory"
        ]
    },
    {
        "id": "Q03",
        "query": "How has the remote work policy changed over time?",
        "category": "TIMELINE_EVOLUTION",
        "expected_behavior": "Router should classify as 'timeline'. Should present both 2022 and 2025 policies chronologically, showing the evolution: 2 days->3-5 days, no stipend->$500-1500, 9-5->flexible hours, VPN->Zero Trust.",
        "expected_facts": [
            "2022: 2 days remote, no stipend, 9-5 hours",
            "2025: 3-5 days by seniority, stipend added, flexible hours",
            "Chronological progression shown"
        ]
    },
    {
        "id": "Q04",
        "query": "What is the daily meal allowance for business travel?",
        "category": "CROSS_MODAL_CONFLICT",
        "expected_behavior": "Should return $100/day from the 2026 structured table (travel_policy_2026.md), NOT $50/day from the 2022 unstructured text (travel_memo_2022.md). The Skeptic should detect the text-vs-table cross-modal conflict and invalidate the 2022 memo.",
        "expected_facts": [
            "Daily meal allowance is $100.00",
            "Receipt required only over $25",
            "2022 cap of $50/day is explicitly superseded",
            "Active policy status from 2026 table"
        ]
    },
    {
        "id": "Q05",
        "query": "What is the base salary for an entry-level software engineer?",
        "category": "CROSS_MODAL_CONFLICT",
        "expected_behavior": "Should return $105,000-$120,000 from the 2026 table, NOT $85,000 from the 2023 text. The Skeptic should flag the 2023 text as superseded by the 2026 structured table.",
        "expected_facts": [
            "Entry engineer (L3) salary is $105,000 - $120,000",
            "Annual bonus is 12%",
            "RSU grant of $20,000/yr",
            "2023 figure of $85,000 is outdated"
        ]
    },
    {
        "id": "Q06",
        "query": "How long does the company retain customer personal data?",
        "category": "CROSS_MODAL_CONFLICT",
        "expected_behavior": "Should return 3 years maximum from the 2025 GDPR-aligned table, NOT 'indefinitely' from the 2021 text. The Skeptic should detect the massive contradiction (indefinite vs 3 years) and invalidate the 2021 doc.",
        "expected_facts": [
            "Customer PII retained for 3 years maximum",
            "Cloud storage now permitted with AES-256 encryption",
            "2021 policy of indefinite retention is superseded",
            "Data access requests fulfilled within 30 days"
        ]
    },
    {
        "id": "Q07",
        "query": "Is there currently a hiring freeze?",
        "category": "CURRENT_STATE",
        "expected_behavior": "Should determine that the 2023 hiring freeze was lifted in February 2024. The answer should state NO, there is no current hiring freeze. The Skeptic should flag the 2023 freeze notice as superseded by the 2024 resumption notice.",
        "expected_facts": [
            "No current hiring freeze",
            "Freeze was lifted in February 2024",
            "Normal recruitment has resumed",
            "New requisitions need Director-level approval"
        ]
    },
    {
        "id": "Q08",
        "query": "What is the performance review rating scale used by the company?",
        "category": "SINGLE_DOC_RETRIEVAL",
        "expected_behavior": "Should retrieve the 2024 performance review policy accurately. No temporal conflict expected since there's only one version. Tests baseline retrieval accuracy.",
        "expected_facts": [
            "5-point scale",
            "Exceeds Expectations, Meets Expectations, Developing, Below Expectations, Unsatisfactory",
            "Annual review in December",
            "PIP for two consecutive Below Expectations ratings"
        ]
    },
    {
        "id": "Q09",
        "query": "What was the lodging reimbursement limit in 2022 and what is it now?",
        "category": "TIMELINE_EVOLUTION",
        "expected_behavior": "Router should classify as 'timeline'. Should compare: 2022 was $200/night (flat), 2026 is $250 standard / $350 major city (tiered). Should show the evolution clearly.",
        "expected_facts": [
            "2022: $200/night flat limit",
            "2026: $250 standard, $350 major city",
            "Tier-based system replaced flat rate",
            "Both time periods addressed"
        ]
    },
    {
        "id": "Q10",
        "query": "Is cloud storage allowed for customer data?",
        "category": "CROSS_MODAL_CONFLICT",
        "expected_behavior": "Should return YES from the 2025 policy (cloud with AES-256), NOT 'No cloud storage permitted' from the 2021 policy. The Skeptic should detect the direct contradiction between 2021 (cloud prohibited) and 2025 (cloud permitted with encryption).",
        "expected_facts": [
            "Cloud storage IS now permitted",
            "AES-256 encryption is mandatory",
            "2021 prohibition of cloud storage is superseded",
            "Effective May 2025"
        ]
    }
]


def run_test(test_case: dict) -> dict:
    """Execute a single test query and capture all outputs."""
    query_id = test_case["id"]
    query = test_case["query"]
    
    print(f"\n{'─' * 60}")
    print(f"  Running {query_id}: {query}")
    print(f"{'─' * 60}")
    
    result = {
        "query_id": query_id,
        "query": query,
        "category": test_case["category"],
        "expected_behavior": test_case["expected_behavior"],
        "expected_facts": test_case["expected_facts"],
        "status": "SUCCESS",
        "error": None,
        "execution_time_seconds": 0,
        "pipeline_output": None
    }
    
    start = time.time()
    try:
        pipeline_output = execute_chronorag_pipeline(query)
        result["execution_time_seconds"] = round(time.time() - start, 2)
        
        # Serialize the output
        result["pipeline_output"] = {
            "final_answer": pipeline_output["answer"],
            "router": pipeline_output["router"],
            "conflict_report": pipeline_output["conflict_report"],
            "retrieved_chunks": [
                {
                    "id": c["id"],
                    "effective_date": c["metadata"].get("effective_date", "N/A"),
                    "doc_type": c["metadata"].get("doc_type", "N/A"),
                    "title": c["metadata"].get("title", "N/A"),
                    "content_preview": c["document"][:300]
                }
                for c in pipeline_output["retrieved_chunks"]
            ]
        }
        
        print(f"  [OK] Completed in {result['execution_time_seconds']}s")
        print(f"    Router Intent: {pipeline_output['router'].get('intent_type', 'N/A')}")
        print(f"    Conflicts Found: {pipeline_output['conflict_report'].get('has_conflicts', False)}")
        print(f"    Chunks Retrieved: {len(pipeline_output['retrieved_chunks'])}")
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["execution_time_seconds"] = round(time.time() - start, 2)
        print(f"  [FAIL] FAILED: {result['error']}")
        traceback.print_exc()
    
    return result


def main():
    print("\n" + "=" * 60)
    print("  ChronoRAG Pipeline Test Suite")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for test in TEST_QUERIES:
        result = run_test(test)
        results.append(result)
        # Small delay to avoid Groq rate limits
        time.sleep(1)
    
    total_time = round(time.time() - total_start, 2)
    
    # Summary
    successes = sum(1 for r in results if r["status"] == "SUCCESS")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    
    report = {
        "test_run_timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful": successes,
        "errors": errors,
        "total_execution_time_seconds": total_time,
        "results": results
    }
    
    # Write report to file
    report_path = "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"  TEST SUITE COMPLETE")
    print(f"  Success: {successes}/10 | Errors: {errors}/10")
    print(f"  Total Time: {total_time}s")
    print(f"  Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
