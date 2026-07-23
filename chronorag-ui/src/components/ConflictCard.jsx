function Stamp({ variant, children }) {
  const colors =
    variant === "invalid"
      ? "border-invalid/70 text-invalid"
      : variant === "verified"
      ? "border-verified/70 text-verified"
      : "border-flagged/70 text-flagged";
  return (
    <span
      className={`inline-block rotate-[-6deg] rounded-sm border-[1.5px] px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.15em] ${colors} animate-stampIn`}
    >
      {children}
    </span>
  );
}

function DocChip({ id, date, invalidated }) {
  return (
    <div className={`flex flex-col gap-1 rounded-md border border-hairline bg-ink/60 px-3 py-2 ${invalidated ? "opacity-70" : ""}`}>
      <span className={`font-mono text-xs ${invalidated ? "text-muted line-through decoration-invalid/70" : "text-text"}`}>
        {id}
      </span>
      <span className="font-mono text-[10px] text-muted">{date || "no date"}</span>
    </div>
  );
}

export default function ConflictCard({ conflict, index, docDates = {} }) {
  return (
    <div
      className="animate-riseIn rounded-lg border border-hairline bg-panel2 p-4"
      style={{ animationDelay: `${index * 60}ms`, animationFillMode: "backwards" }}
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
          Conflict Entry #{String(index + 1).padStart(2, "0")}
        </span>
        <Stamp variant="invalid">Superseded</Stamp>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <DocChip id={conflict.invalidated_doc_id} date={docDates[conflict.invalidated_doc_id]} invalidated />
        <span className="font-mono text-lg text-stamp">&rarr;</span>
        <DocChip id={conflict.valid_doc_id} date={docDates[conflict.valid_doc_id]} />
        <Stamp variant="verified">Current</Stamp>
      </div>

      <p className="mt-3 border-t border-hairline pt-3 text-[13px] leading-relaxed text-muted">
        {conflict.conflict_reason}
      </p>
    </div>
  );
}
