// Demo fixture shaped exactly like execute_chronorag_pipeline()'s return value,
// so the UI renders meaningfully before the FastAPI bridge is wired up.
export const MOCK_RESULT = {
  router: {
    intent_type: "current",
    target_year: null,
    reasoning: "Query asks for the active policy limit with no date qualifier, implying present-day state.",
  },
  retrieved_chunks: [
    {
      id: "LOCAL-remote_work_2022.md-chunk-0",
      document:
        "Remote work reimbursement is capped at $500 per quarter for home office equipment, effective for all full-time staff.",
      metadata: { effective_date: "2022-03-01", doc_type: "unstructured_text", source: "local_file" },
    },
    {
      id: "LOCAL-remote_work_2026.md-chunk-2",
      document:
        "| Category | Limit | Status |\n| Home Office Equipment | $1,200 / quarter | Active |\n| Internet Stipend | $60 / month | Active |",
      metadata: { effective_date: "2026-01-15", doc_type: "structured_table", source: "local_file" },
    },
    {
      id: "TLAMA-118",
      document: "Historical Policy Note (2019): Regarding 'remote work stipend', official stance is $200.",
      metadata: { effective_date: "2019-01-01", doc_type: "unstructured_text", source: "templama" },
    },
  ],
  conflict_report: {
    has_conflicts: true,
    confidence_score: 0.91,
    conflicts: [
      {
        invalidated_doc_id: "LOCAL-remote_work_2022.md-chunk-0",
        valid_doc_id: "LOCAL-remote_work_2026.md-chunk-2",
        conflict_reason:
          "The 2026 structured table raises the quarterly equipment cap from $500 to $1,200, explicitly superseding the 2022 text policy on the same topic.",
      },
    ],
    surviving_doc_ids: ["LOCAL-remote_work_2026.md-chunk-2"],
  },
  answer:
    "**Current remote work equipment limit: $1,200 per quarter.**\n\nThis reflects the active 2026 policy table (`LOCAL-remote_work_2026.md-chunk-2`), which supersedes the earlier 2022 text policy that capped reimbursement at $500/quarter. A monthly internet stipend of $60 is also active under the same schedule.\n\n*Note: a 2019 historical reference ($200) was excluded as stale and unrelated to the current query.*",
};

export const SAMPLE_QUERIES = [
  "What is the remote work policy limit?",
  "How has the remote work policy changed over time?",
  "What was the reimbursement policy in 2022?",
];
