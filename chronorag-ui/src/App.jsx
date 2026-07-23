import { useState } from "react";
import Cursor from "./components/Cursor.jsx";
import Stepper from "./components/Stepper.jsx";
import ConflictCard from "./components/ConflictCard.jsx";
import DocCard from "./components/DocCard.jsx";
import { runPipeline } from "./api.js";
import { SAMPLE_QUERIES } from "./mock.js";

const INTENT_COPY = {
  current: { label: "Current State", color: "text-verified", note: "Showing the latest verified policy." },
  historical: { label: "Historical Snapshot", color: "text-flagged", note: "Showing the policy as it stood at the requested time." },
  timeline: { label: "Timeline / Evolution", color: "text-stamp", note: "Showing how this policy changed over time." },
};

export default function App() {
  const [query, setQuery] = useState("What is the remote work policy limit?");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(-1);
  const [selectedTab, setSelectedTab] = useState("skeptic");
  const [result, setResult] = useState(null);
  const [live, setLive] = useState(false);

  const execute = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setStage(0);

    // Stage the reveal so the pipeline reads as sequential work,
    // matching what's actually happening server-side.
    const t1 = setTimeout(() => setStage(1), 350);
    const t2 = setTimeout(() => setStage(2), 700);

    const { data, live: isLive } = await runPipeline(query);

    clearTimeout(t1);
    clearTimeout(t2);
    setStage(3);
    setResult(data);
    setLive(isLive);
    setSelectedTab("skeptic");
    setLoading(false);
  };

  const docDates = {};
  result?.retrieved_chunks?.forEach((c) => (docDates[c.id] = c.metadata?.effective_date));
  const survivingIds = new Set(result?.conflict_report?.surviving_doc_ids || []);
  const intent = result?.router?.intent_type ? INTENT_COPY[result.router.intent_type] : null;

  return (
    <div className="min-h-screen">
      <Cursor />

      {/* Header */}
      <header className="border-b border-hairline">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <div className="flex items-baseline gap-3">
            <h1 className="font-display text-lg font-semibold tracking-tight">ChronoRAG</h1>
            <span className="font-mono text-[11px] text-muted">Temporal Reconciliation Console</span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px] text-muted">
            <span className={`h-1.5 w-1.5 rounded-full ${live ? "bg-verified" : "bg-flagged"}`} />
            {result ? (live ? "Live pipeline" : "Demo fixture") : "Idle"}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {/* Query intake */}
        <section className="mb-8 rounded-xl border border-hairline bg-panel p-5">
          <label className="mb-2 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
            Case Query
          </label>
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && execute()}
              placeholder="Ask about a policy, rule, or figure that may have changed over time…"
              className="flex-1 rounded-lg border border-hairline bg-ink px-4 py-3 font-body text-[14px] text-text placeholder:text-muted/70 focus:border-stamp focus:outline-none"
            />
            <button
              data-cursor="stamp"
              onClick={execute}
              disabled={loading}
              className="rounded-lg bg-stamp px-6 py-3 font-display text-[13px] font-semibold tracking-tight text-ink transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {loading ? "Auditing…" : "Run Pipeline"}
            </button>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {SAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                data-cursor="stamp"
                onClick={() => setQuery(q)}
                className="rounded-full border border-hairline px-3 py-1 font-mono text-[10.5px] text-muted hover:border-stamp hover:text-text"
              >
                {q}
              </button>
            ))}
          </div>
        </section>

        {/* Stepper */}
        <section className="mb-8 overflow-hidden rounded-xl border border-hairline">
          <Stepper activeIndex={stage} onSelect={() => {}} selected={null} />
        </section>

        {!result && !loading && (
          <div className="rounded-xl border border-dashed border-hairline py-20 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">No case run yet</p>
            <p className="mt-2 text-sm text-muted">Enter a query above and run the pipeline to see the audit trail.</p>
          </div>
        )}

        {result && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            {/* Answer panel */}
            <section className="animate-riseIn rounded-xl border border-hairline bg-panel p-6 lg:col-span-3">
              <div className="mb-4 flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
                  Audited Response
                </span>
                {intent && (
                  <span className={`font-mono text-[10px] uppercase tracking-[0.15em] ${intent.color}`}>
                    {intent.label}
                  </span>
                )}
              </div>
              {intent && <p className="mb-4 text-[11px] text-muted">{intent.note}</p>}
              <div className="whitespace-pre-wrap font-body text-[14.5px] leading-relaxed text-text">
                {result.answer}
              </div>
            </section>

            {/* Trace panel */}
            <section className="lg:col-span-2">
              <div className="mb-3 flex gap-1 rounded-lg border border-hairline bg-panel p-1">
                {[
                  { id: "skeptic", label: "Conflict Ledger" },
                  { id: "evidence", label: "Evidence" },
                  { id: "router", label: "Router" },
                ].map((t) => (
                  <button
                    key={t.id}
                    data-cursor="stamp"
                    onClick={() => setSelectedTab(t.id)}
                    className={`flex-1 rounded-md px-2 py-2 font-mono text-[10.5px] uppercase tracking-wider transition-colors ${
                      selectedTab === t.id ? "bg-panel2 text-text" : "text-muted hover:text-text"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              <div className="max-h-[560px] space-y-3 overflow-y-auto scrollbar-thin pr-1">
                {selectedTab === "skeptic" && (
                  <>
                    <div className="flex items-center justify-between rounded-lg border border-hairline bg-panel p-3">
                      <span className="font-mono text-[10px] text-muted">Confidence</span>
                      <span className="font-mono text-[12px] text-verified">
                        {(result.conflict_report.confidence_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    {result.conflict_report.conflicts.length === 0 ? (
                      <div className="rounded-lg border border-hairline bg-panel p-4 text-center font-mono text-[11px] text-muted">
                        No conflicts detected — all evidence agrees.
                      </div>
                    ) : (
                      result.conflict_report.conflicts.map((c, i) => (
                        <ConflictCard key={i} conflict={c} index={i} docDates={docDates} />
                      ))
                    )}
                  </>
                )}

                {selectedTab === "evidence" &&
                  result.retrieved_chunks.map((doc, i) => (
                    <DocCard key={doc.id} doc={doc} survived={survivingIds.has(doc.id)} index={i} />
                  ))}

                {selectedTab === "router" && (
                  <div className="space-y-3 rounded-lg border border-hairline bg-panel p-4 font-mono text-[12px]">
                    <div className="flex justify-between">
                      <span className="text-muted">intent_type</span>
                      <span className="text-text">{result.router.intent_type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">target_year</span>
                      <span className="text-text">{result.router.target_year || "—"}</span>
                    </div>
                    <div className="border-t border-hairline pt-3 text-[11.5px] leading-relaxed text-muted">
                      {result.router.reasoning}
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-6 py-10 font-mono text-[10px] text-muted">
        ChronoRAG · Multi-agent temporal reconciliation over Groq (Llama 3.3 70B) + ChromaDB
      </footer>
    </div>
  );
}
