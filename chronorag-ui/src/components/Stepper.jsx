const STAGES = [
  { id: "router", n: "01", label: "Temporal Router", detail: "Classify intent" },
  { id: "retrieval", n: "02", label: "Hybrid Retrieval", detail: "Dense + BM25 · RRF" },
  { id: "skeptic", n: "03", label: "Reconciliation Skeptic", detail: "Audit for conflicts" },
  { id: "synthesis", n: "04", label: "Synthesis", detail: "Grounded response" },
];

export default function Stepper({ activeIndex, onSelect, selected }) {
  return (
    <div className="flex w-full items-stretch gap-0 overflow-x-auto scrollbar-thin">
      {STAGES.map((s, i) => {
        const done = i <= activeIndex;
        const isSelected = selected === s.id;
        return (
          <button
            key={s.id}
            data-cursor="stamp"
            disabled={!done}
            onClick={() => done && onSelect(s.id)}
            className={`group relative flex min-w-[190px] flex-1 items-center gap-3 border-b-2 px-4 py-3 text-left transition-colors
              ${isSelected ? "border-stamp bg-panel2" : "border-hairline/70 hover:border-hairline"}
              ${done ? "opacity-100" : "opacity-35 cursor-default"}`}
          >
            <span
              className={`font-mono text-[11px] leading-none ${
                done ? (isSelected ? "text-stamp" : "text-verified") : "text-muted"
              }`}
            >
              {s.n}
            </span>
            <span className="flex flex-col">
              <span className="font-display text-[13px] font-medium tracking-tight text-text">
                {s.label}
              </span>
              <span className="font-mono text-[10px] text-muted">{s.detail}</span>
            </span>
            {i === activeIndex && (
              <span className="absolute right-3 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-verified animate-pulseDot" />
            )}
          </button>
        );
      })}
    </div>
  );
}
