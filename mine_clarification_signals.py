"""
Clarification Gate Tuning — find dev cases where the deterministic gate
missed (`clarification_reasons` empty but `true_needs_clarification=True`),
have an LLM open-code each miss into a short reason phrase, then cluster
the codes to suggest new deterministic features.

Usage:
    python3 mine_clarification_signals.py

Outputs:
    models/candidate_clarification_signals.json — clusters + codes + examples

Adoption: human reads the report, picks coherent clusters, manually writes
a new `_feature_name(turns)` helper in agent.py:234-ish, adds it to
decide_clarification at agent.py:299, re-runs evaluation/run_eval.py to
confirm clarification F1 improves.

Note: clusters are produced via cosine-similarity grouping over embeddings
of the LLM-generated codes — no sklearn dependency. Clustering is
deliberately coarse since the gate misses are typically small (~5-20 cases).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "evaluation"
MODELS = HERE / "models"

PREDICTIONS = HERE / "predictions.json"
DEV_TRANSCRIPTS = EVAL_DIR / "eval_transcripts_dev.json"
DEV_LABELS = EVAL_DIR / "dev_labels.json"
REPORT = MODELS / "candidate_clarification_signals.json"

CODE_PROMPT = """A human reviewer flagged the following maintenance call as needing
clarification (`true_needs_clarification=True`), but our deterministic gate
failed to flag it. The gate currently catches four signals:
  1. sparse_caller       — caller's total speech <= 35 words
  2. taxonomy_echo       — caller parroted a subcategory label
  3. floor_self_correction — caller named two different floors
  4. floor_exceeds_building — claimed floor > building.floor_count

In ONE short phrase (5-10 words), describe the signal the gate should have
caught. Be concrete — name the structural pattern, not the topic.
Examples of good codes:
  - "caller mentioned two different buildings"
  - "symptom changed mid-call from gas to smoke"
  - "caller used pronoun without antecedent for location"
  - "conflicting urgency cues across turns"

TRANSCRIPT:
{transcript}

Phrase:"""

CODE_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

# Cluster threshold: codes with cosine similarity >= this go into the same
# cluster. Conservative so small clusters surface distinct patterns.
SIM_THRESHOLD = 0.55


def _load_data() -> tuple[list[dict], dict, dict]:
    preds = json.loads(PREDICTIONS.read_text())
    labels = json.loads(DEV_LABELS.read_text())["by_id"]
    transcripts = {
        r["transcript_id"]: r for r in json.loads(DEV_TRANSCRIPTS.read_text())
    }
    return preds, labels, transcripts


def _flatten_turns(turns: list[dict]) -> str:
    return "\n".join(f"[{t['speaker'].upper()}] {t['text']}" for t in turns)


def _find_misses(preds: list[dict], labels: dict, transcripts: dict) -> list[dict]:
    misses = []
    for pred in preds:
        tid = pred["transcript_id"]
        label = labels.get(tid)
        if not label or not label.get("true_needs_clarification"):
            continue
        tlog = pred.get("trainer_log") or {}
        reasons = (tlog.get("ai_prediction") or {}).get("clarification_reasons") or []
        if reasons:
            # Gate fired — not a miss.
            continue
        tx = transcripts.get(tid)
        if not tx:
            continue
        misses.append({
            "transcript_id": tid,
            "transcript": _flatten_turns(tx["turns"]),
            "true_subcategory": label.get("true_subcategory"),
            "true_risk_level": label.get("true_risk_level"),
        })
    return misses


def _open_code(misses: list[dict], llm: ChatOpenAI) -> None:
    """In-place: add `code` field per miss."""
    for m in misses:
        prompt = CODE_PROMPT.format(transcript=m["transcript"])
        try:
            resp = llm.invoke([{"role": "user", "content": prompt}])
            code = (resp.content or "").strip().strip(".").strip('"').strip("'")
        except Exception as e:
            code = f"ERROR: {e}"
        m["code"] = code


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _cluster(misses: list[dict], embedder: OpenAIEmbeddings) -> list[dict]:
    """Greedy single-link cosine clustering. No sklearn."""
    if not misses:
        return []
    codes = [m["code"] for m in misses]
    vecs = embedder.embed_documents(codes)

    cluster_ids = [-1] * len(misses)
    next_id = 0
    for i in range(len(misses)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, len(misses)):
            if cluster_ids[j] != -1:
                continue
            if _cosine(vecs[i], vecs[j]) >= SIM_THRESHOLD:
                cluster_ids[j] = next_id
        next_id += 1

    clusters: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(idx)

    out = []
    for cid, member_idxs in sorted(clusters.items(), key=lambda x: -len(x[1])):
        members = [misses[i] for i in member_idxs]
        # Representative code = the one with highest mean cosine to others.
        if len(member_idxs) == 1:
            rep_idx = member_idxs[0]
        else:
            best_idx, best_score = member_idxs[0], -1.0
            for i in member_idxs:
                s = sum(_cosine(vecs[i], vecs[j]) for j in member_idxs if j != i)
                if s > best_score:
                    best_score = s
                    best_idx = i
            rep_idx = best_idx
        out.append({
            "cluster_id": cid,
            "size": len(members),
            "representative_code": misses[rep_idx]["code"],
            "all_codes": [m["code"] for m in members],
            "example_transcripts": [
                {
                    "transcript_id": m["transcript_id"],
                    "true_subcategory": m["true_subcategory"],
                    "true_risk_level": m["true_risk_level"],
                    "code": m["code"],
                    "transcript": m["transcript"],
                }
                for m in members[:3]
            ],
        })
    return out


def main() -> None:
    MODELS.mkdir(exist_ok=True)

    print("Loading predictions, labels, transcripts ...")
    preds, labels, transcripts = _load_data()

    n_pos = sum(1 for lab in labels.values() if lab.get("true_needs_clarification"))
    print(f"  ground-truth clarification positives: {n_pos}")

    misses = _find_misses(preds, labels, transcripts)
    print(f"  gate misses (gate=False, truth=True): {len(misses)}")

    if not misses:
        REPORT.write_text(json.dumps({
            "gate_recall_current": 1.0,
            "n_misses": 0,
            "message": "Gate caught every true positive — no clusters to mine.",
            "clusters": [],
        }, indent=2))
        print(f"\nNo misses. Wrote empty report → {REPORT}")
        return

    gate_recall = (n_pos - len(misses)) / n_pos if n_pos else 1.0
    print(f"  gate recall on dev: {gate_recall:.2%}")

    print("\nOpen-coding each miss via LLM ...")
    llm = ChatOpenAI(model=CODE_MODEL, temperature=0, max_retries=3, request_timeout=20)
    _open_code(misses, llm)
    for m in misses:
        print(f"  [{m['transcript_id']}] {m['code']}")

    print("\nEmbedding codes and clustering ...")
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    clusters = _cluster(misses, embedder)

    code_counter = Counter(m["code"] for m in misses)

    report = {
        "n_clarification_positives": n_pos,
        "n_misses": len(misses),
        "gate_recall_current": round(gate_recall, 4),
        "cluster_similarity_threshold": SIM_THRESHOLD,
        "all_codes_frequency": dict(code_counter.most_common()),
        "clusters": clusters,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")
    print(f"\n{len(clusters)} cluster(s):")
    for c in clusters:
        print(f"  [cluster {c['cluster_id']}] size={c['size']}  "
              f"rep=\"{c['representative_code']}\"")


if __name__ == "__main__":
    main()
