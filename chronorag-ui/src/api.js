import { MOCK_RESULT } from "./mock.js";

// Your Python side is currently a Streamlit app (app.py), which has no HTTP
// endpoint a browser can call. The lightest bridge is a ~10-line FastAPI
// wrapper around execute_chronorag_pipeline() from pipeline.py, e.g.:
//
//   from fastapi import FastAPI
//   from pipeline import execute_chronorag_pipeline
//   app = FastAPI()
//   @app.post("/api/query")
//   def query(payload: dict):
//       return execute_chronorag_pipeline(payload["query"])
//
// Run it with `uvicorn server:app --port 8000` and the Vite dev proxy
// (see vite.config.js) will route /api/query straight to it.
export async function runPipeline(query) {
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) throw new Error(`Backend responded ${res.status}`);
    return { data: await res.json(), live: true };
  } catch (err) {
    // No backend reachable yet — fall back to the fixture so the
    // interface stays fully demonstrable.
    await new Promise((r) => setTimeout(r, 650));
    return { data: MOCK_RESULT, live: false };
  }
}
