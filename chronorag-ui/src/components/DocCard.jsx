export default function DocCard({ doc, survived, index }) {
  const isTable = doc.metadata?.doc_type === "structured_table";
  return (
    <div
      className={`animate-riseIn rounded-lg border bg-panel2 p-3.5 transition-colors ${
        survived ? "border-hairline" : "border-hairline/50 opacity-55"
      }`}
      style={{ animationDelay: `${index * 40}ms`, animationFillMode: "backwards" }}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-muted">{isTable ? "TBL" : "TXT"}</span>
          <span className="font-mono text-xs text-text">{doc.id}</span>
        </div>
        <span
          className={`rounded-full px-2 py-0.5 font-mono text-[9px] uppercase tracking-widest ${
            survived ? "bg-verified/10 text-verified" : "bg-invalid/10 text-invalid"
          }`}
        >
          {survived ? "Verified" : "Invalidated"}
        </span>
      </div>
      <p className="line-clamp-3 text-[12.5px] leading-relaxed text-muted">{doc.document}</p>
      <div className="mt-2 flex items-center gap-3 border-t border-hairline pt-2 font-mono text-[10px] text-muted">
        <span>{doc.metadata?.effective_date || "—"}</span>
        <span className="text-hairline">·</span>
        <span>{doc.metadata?.source || "unknown"}</span>
      </div>
    </div>
  );
}
