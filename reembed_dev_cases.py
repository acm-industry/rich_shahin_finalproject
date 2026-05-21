"""
RAG Corpus Refresh — embed dev cases (with ground-truth labels) into a
shadow Chroma corpus, then run dev eval against it to verify no regression
before promotion.

Usage:
    python3 reembed_dev_cases.py

Outputs:
    chroma_db_shadow/                 — historical corpus + dev cases
    predictions_baseline.json         — eval against current prod corpus
    predictions_shadow.json           — eval against shadow corpus
    models/reembed_report.json        — composite delta + per-axis breakdown

Adoption: if the report shows composite improvement and false_911_count
does not increase, rename chroma_db_shadow → chroma_db (after backing up).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "evaluation"
MODELS = HERE / "models"
PROD_CHROMA = HERE / "chroma_db"
SHADOW_CHROMA = HERE / "chroma_db_shadow"
COLLECTION = "historical_records"

DEV_TRANSCRIPTS = EVAL_DIR / "eval_transcripts_dev.json"
DEV_LABELS = EVAL_DIR / "dev_labels.json"

BASELINE_PREDS = HERE / "predictions_baseline.json"
SHADOW_PREDS = HERE / "predictions_shadow.json"
REPORT = MODELS / "reembed_report.json"


def _flatten_turns(turns: list[dict]) -> str:
    return " ".join(t.get("text", "") for t in turns if t.get("speaker") == "caller")


def _build_dev_documents() -> list[Document]:
    transcripts = json.loads(DEV_TRANSCRIPTS.read_text())
    labels = json.loads(DEV_LABELS.read_text())["by_id"]
    docs: list[Document] = []
    for row in transcripts:
        tid = row["transcript_id"]
        label = labels.get(tid)
        if not label:
            continue
        # Mirror build_chroma.py page_content schema so retrieval treats these
        # the same way as historical_records entries.
        text = (
            f"Problem: {_flatten_turns(row['turns'])}\n"
            f"Category: {label.get('true_category','')} / {label.get('true_subcategory','')}\n"
            f"Risk: {label.get('true_risk_level','')}\n"
            f"Building: {label.get('ground_truth_building_name','')} "
            f"({label.get('ground_truth_building_type','')}), "
            f"Floor: {label.get('ground_truth_floor','')}"
        )
        metadata = {
            "ticket_id": f"DEV-{tid}",
            "final_category": label.get("true_category") or "",
            "final_subcategory": label.get("true_subcategory") or "",
            "final_risk_level": label.get("true_risk_level") or "",
            "building_name": label.get("ground_truth_building_name") or "",
            "city": label.get("ground_truth_city") or "",
            "floor": label.get("ground_truth_floor") or "",
            "assigned_vendor_id": (label.get("acceptable_vendor_ids") or [""])[0],
            "source": "dev_validated",
            "_tier": "A_dev_validated",
            "_weight": 1.0,
            # Match build_chroma.py audit-flag schema so consensus_check
            # doesn't choke on missing keys.
            "was_over_escalated": "false",
            "was_reclassified": "false",
            "floor_was_wrong_at_intake": "false",
            "audit_clean": "true",
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def _build_shadow_corpus(force: bool = False) -> int:
    if SHADOW_CHROMA.exists() and force:
        shutil.rmtree(SHADOW_CHROMA)
    if not SHADOW_CHROMA.exists():
        print(f"Copying {PROD_CHROMA} → {SHADOW_CHROMA} ...")
        shutil.copytree(PROD_CHROMA, SHADOW_CHROMA)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(SHADOW_CHROMA),
    )
    n_before = db._collection.count()

    docs = _build_dev_documents()
    # Don't double-append if script is re-run on the same shadow corpus.
    new_ids = [d.metadata["ticket_id"] for d in docs]
    existing = db._collection.get(ids=new_ids)
    already_in = set(existing["ids"]) if existing.get("ids") else set()
    fresh = [d for d in docs if d.metadata["ticket_id"] not in already_in]

    if fresh:
        print(f"Embedding {len(fresh)} dev cases (skipped {len(docs)-len(fresh)} already present) ...")
        db.add_documents(fresh, ids=[d.metadata["ticket_id"] for d in fresh])
    else:
        print(f"All {len(docs)} dev cases already in shadow corpus.")

    n_after = db._collection.count()
    print(f"Shadow corpus size: {n_before} → {n_after}")
    return n_after


def _run_eval(out_path: Path, chroma_dir: Path | None = None) -> None:
    cmd = [
        sys.executable, str(EVAL_DIR / "run_eval.py"),
        "--agent", "agent:classify",
        "--out", str(out_path),
    ]
    if chroma_dir is not None:
        cmd += ["--chroma", str(chroma_dir)]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(HERE))


def _score(predictions_path: Path) -> dict:
    """Import scoring.grade and return composite + breakdown."""
    sys.path.insert(0, str(EVAL_DIR))
    from scoring import grade, index_preds, load_json  # type: ignore

    eval_rows = load_json(DEV_TRANSCRIPTS)
    gt_raw = load_json(DEV_LABELS)
    gt_map = gt_raw.get("by_id", gt_raw)
    preds = load_json(predictions_path)
    preds_map = index_preds(preds)
    return grade(preds_map, eval_rows, gt_map)


def main():
    MODELS.mkdir(exist_ok=True)

    # 1. Baseline eval (only if not already cached)
    if not BASELINE_PREDS.exists():
        print("=== Step 1: baseline eval (prod corpus) ===")
        _run_eval(BASELINE_PREDS, chroma_dir=None)
    else:
        print(f"Reusing existing baseline: {BASELINE_PREDS}")

    # 2. Build shadow corpus
    print("\n=== Step 2: build shadow corpus ===")
    shadow_count = _build_shadow_corpus()

    # 3. Shadow eval
    print("\n=== Step 3: shadow eval ===")
    _run_eval(SHADOW_PREDS, chroma_dir=SHADOW_CHROMA)

    # 4. Score both
    print("\n=== Step 4: score and compare ===")
    baseline = _score(BASELINE_PREDS)
    shadow = _score(SHADOW_PREDS)

    report = {
        "shadow_corpus_size": shadow_count,
        "baseline": {
            "composite": baseline["composite"],
            "composite_raw": baseline["composite_raw"],
            "false_911_count": baseline["false_911_count"],
            "axis_pct": baseline["axis_pct"],
            "hitl_f1": baseline["hitl"]["f1"],
            "clarification_f1": baseline["clarification"]["f1"],
        },
        "shadow": {
            "composite": shadow["composite"],
            "composite_raw": shadow["composite_raw"],
            "false_911_count": shadow["false_911_count"],
            "axis_pct": shadow["axis_pct"],
            "hitl_f1": shadow["hitl"]["f1"],
            "clarification_f1": shadow["clarification"]["f1"],
        },
        "delta": {
            "composite": round(shadow["composite"] - baseline["composite"], 4),
            "false_911_delta": shadow["false_911_count"] - baseline["false_911_count"],
        },
        "recommendation": (
            "ADOPT"
            if shadow["composite"] > baseline["composite"]
            and shadow["false_911_count"] <= baseline["false_911_count"]
            else "REVIEW"
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")
    print(f"  baseline composite: {baseline['composite']:.2f}")
    print(f"  shadow composite:   {shadow['composite']:.2f}")
    print(f"  delta:              {report['delta']['composite']:+.4f}")
    print(f"  false_911_delta:    {report['delta']['false_911_delta']:+d}")
    print(f"  recommendation:     {report['recommendation']}")


if __name__ == "__main__":
    main()
