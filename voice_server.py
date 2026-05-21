"""Voice demo server.

Wraps the existing `classify()` agent behind a tiny HTTP shim so a browser
frontend can drive the agent with Web Speech API STT/TTS.

The agent contract is untouched — this server builds the same
`[{"speaker": ..., "text": ...}]` shape and hands it to the same graph used
by `classify()`. The only extra it exposes is `clarification_question`,
which already lives in graph state but is dropped by `_format_prediction()`.

Run:
    python voice_server.py
Then open http://localhost:8000/ in Chrome or Edge.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from agent import (
    _flatten,
    _format_prediction,
    _get_eval_graph,
    _lookup_profile,
    make_extract_and_check,
)

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
INDEX_FILE = STATIC_DIR / "index.html"

PORT = 8000

# Voice-mode gating thresholds
MIN_CALLER_WORDS = 8        # below this, ALWAYS ask for more — too sparse to classify
MAX_FOLLOWUP_TURNS = 4      # after this many follow-ups, give up and escalate to HITL

# Standalone extractor for the voice-mode pre-check. Same LLM the graph uses.
_LLM_MINI = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3, request_timeout=20)
_VOICE_EXTRACTOR = make_extract_and_check(_LLM_MINI)


def _caller_word_count(turns: list[dict]) -> int:
    return sum(len(t.get("text", "").split()) for t in turns if t.get("speaker") == "caller")


def _agent_turn_count(turns: list[dict]) -> int:
    """How many follow-up questions has the agent already asked?"""
    return sum(1 for t in turns if t.get("speaker") == "agent") - 1  # minus the initial greeting


def _build_clarification_response(question: str, profile: dict | None) -> dict:
    """Voice-mode payload: ask the question, don't surface a (partial) classification."""
    return {
        "category": None,
        "subcategory": None,
        "risk_level": None,
        "needs_human_review": False,
        "needs_clarification": True,
        "clarification_question": question,
        "call_summary": "",
        "building_name": None,
        "floor": None,
        "address": None,
        "dispatched_vendor_id": None,
        "dispatched_emergency_services": False,
        "trainer_log": None,
        "caller_profile": profile,
    }


def classify_voice(turns: list[dict], caller_phone: str | None) -> dict:
    """Voice-side wrapper. Multi-turn aware: returns a clarifying question
    while extraction is incomplete instead of closing the call early.
    Falls back to full classification once enough info accumulates, or
    escalates to HITL after MAX_FOLLOWUP_TURNS questions if still incomplete.
    """
    profile = _lookup_profile(caller_phone)
    transcript = _flatten(turns)

    # ── Voice-mode pre-check: are we ready to classify? ────────────────────
    caller_words = _caller_word_count(turns)
    followups_asked = _agent_turn_count(turns)

    if followups_asked < MAX_FOLLOWUP_TURNS:
        # 1) Hard floor: too few caller words to classify on
        if caller_words < MIN_CALLER_WORDS:
            return _build_clarification_response(
                "Could you tell me a bit more about what's going wrong — "
                "what specifically isn't working, where, and how long it's "
                "been like that?",
                profile,
            )

        # 2) Soft check: ask the extractor — is the description specific
        # enough and the location complete?
        pre_state = {
            "transcript": transcript,
            "caller_profile": profile,
            "extraction_complete": False,
            "needs_clarification": False,
        }
        try:
            extract_out = _VOICE_EXTRACTOR(pre_state)
        except Exception:
            extract_out = {}

        if (
            not extract_out.get("extraction_complete", True)
            and extract_out.get("follow_ups")
        ):
            return _build_clarification_response(
                extract_out["follow_ups"][0],
                profile,
            )

    # ── Otherwise: run the full graph as before ────────────────────────────
    graph = _get_eval_graph()

    result = graph.invoke({
        "transcript": transcript,
        "turns_split": turns,
        "caller_phone": caller_phone,
        "caller_profile": profile,
        "eval_mode": True,
        "entity": None,
        "follow_ups": [],
        "extraction_complete": False,
        "needs_clarification": False,
        "gate_clarification_verdict": False,
        "clarification_question": None,
        "clarification_reasons": (),
        "enriched_query": "",
        "retrieved_docs": [],
        "retrieved_context": "",
        "retrieval_attempts": 0,
        "classification": None,
        "consensus_agreement": None,
        "building": None,
        "building_name": None,
        "floor": None,
        "address": None,
        "_city": None,
        "_building_type": None,
        "vendor_id": None,
        "needs_human_review": False,
        "dispatched_emergency_services": False,
        "call_summary": "",
        "routing_decision": None,
        "human_override": None,
        "trainer_log": None,
        "_grade": None,
    })

    pred = _format_prediction(result)
    classification = result.get("classification") or {}
    pred["clarification_question"] = (
        result.get("clarification_question")
        or classification.get("clarification_question")
    )
    pred["caller_profile"] = profile
    return pred


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        print(f"[{self.address_string()}] {fmt % args}")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # CORS preflight (browser-safe even on same origin)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            if not INDEX_FILE.exists():
                self._send_json(500, {"error": "index.html missing"})
                return
            self._send_file(INDEX_FILE, "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = STATIC_DIR / rel
            if not target.is_file() or STATIC_DIR not in target.resolve().parents:
                self.send_error(404)
                return
            ext = target.suffix.lower()
            ctype = {
                ".css": "text/css",
                ".js": "application/javascript",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
            }.get(ext, "application/octet-stream")
            self._send_file(target, ctype)
            return

        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/classify":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"error": "missing body"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"bad json: {e}"})
            return

        turns = payload.get("turns") or []
        caller_phone = payload.get("caller_phone") or None

        if not isinstance(turns, list) or not turns:
            self._send_json(400, {"error": "turns must be a non-empty list"})
            return
        for t in turns:
            if not isinstance(t, dict) or "speaker" not in t or "text" not in t:
                self._send_json(400, {"error": "each turn needs speaker and text"})
                return

        try:
            result = classify_voice(turns, caller_phone)
        except Exception as exc:  # noqa: BLE001 — surface to client
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, result)


def main() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists():
        raise SystemExit(f"missing {INDEX_FILE} — build the frontend first")

    print(f"Warming graph…")
    _get_eval_graph()  # eager compile so first request isn't slow

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Voice demo listening on http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
