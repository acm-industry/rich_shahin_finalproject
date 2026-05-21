"""
Trap Pattern Mining — find subcategory transitions where the agent
over-escalated on dev cases, extract candidate trigger keywords via
lift analysis, and shadow-eval each candidate to measure dev impact.

Usage:
    # Reads existing predictions.json (produced by run_eval.py)
    python3 mine_trap_patterns.py

    # Force a fresh baseline eval first
    python3 mine_trap_patterns.py --rebuild-baseline

Outputs:
    models/candidate_trap_patterns.json    — proposals + dev deltas + examples

Adoption: human reads the JSON, picks candidates with recommendation
"ADOPT" (composite improved, no new false-911s), manually appends them
to TRAP_PATTERNS in agent.py:656.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "evaluation"
MODELS = HERE / "models"

PREDICTIONS = HERE / "predictions.json"
DEV_TRANSCRIPTS = EVAL_DIR / "eval_transcripts_dev.json"
DEV_LABELS = EVAL_DIR / "dev_labels.json"
REPORT = MODELS / "candidate_trap_patterns.json"

DANGEROUS = {
    "fire_smoke", "gas_chemical", "entrapment",
    "active_threat", "structural", "panel_hazard",
}

# Subcategory → category lookup for proposed trap entries. Mirrors
# CATEGORY_SUBCATEGORIES in agent.py.
SUB_TO_CAT = {
    "pipe_leak": "PLUMBING", "restroom_fixture": "PLUMBING",
    "drainage_backup": "PLUMBING", "roof_leak": "PLUMBING",
    "power_outage": "ELECTRICAL", "panel_hazard": "ELECTRICAL",
    "lighting": "ELECTRICAL", "low_voltage_data": "ELECTRICAL",
    "no_cooling": "HVAC", "no_heating": "HVAC", "air_quality": "HVAC",
    "refrigerant": "HVAC", "controls_bms": "HVAC",
    "entrapment": "ELEVATOR", "malfunction": "ELEVATOR", "minor_issue": "ELEVATOR",
    "door_mechanical": "DOORS_ACCESS", "glass_damage": "DOORS_ACCESS",
    "access_control": "DOORS_ACCESS", "auto_door": "DOORS_ACCESS",
    "gas_chemical": "LIFE_SAFETY", "fire_smoke": "LIFE_SAFETY",
    "structural": "LIFE_SAFETY", "slip_trip": "LIFE_SAFETY",
    "suspicious_person": "SECURITY", "unauthorized_access": "SECURITY",
    "active_threat": "SECURITY",
    "restroom_supplies": "JANITORIAL", "carpet_floor": "JANITORIAL",
    "waste_odor": "JANITORIAL",
    "parking_lighting": "GROUNDS_EXTERIOR",
    "pavement_damage": "GROUNDS_EXTERIOR", "signage_fencing": "GROUNDS_EXTERIOR",
    "infestation": "PEST_SPECIALTY", "appliance_kitchen": "PEST_SPECIALTY",
    "landscaping": "PEST_SPECIALTY",
}

# Min-band cap by safer subcategory. Conservative — pick the lowest
# plausible band so the trap only ever demotes.
SAFER_MAX_RISK = {
    "air_quality": "LOW", "waste_odor": "LOW", "malfunction": "MEDIUM",
    "minor_issue": "LOW", "suspicious_person": "MEDIUM", "roof_leak": "MEDIUM",
}

# Tokenization & stop words — small hand-picked list, no sklearn dependency.
STOP_WORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with",
    "is","it","its","was","were","be","been","being","this","that","these","those",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","our",
    "from","by","as","so","just","not","no","yes","do","does","did","have","has","had",
    "can","could","would","should","will","may","might","there","here","what","where",
    "when","who","how","why","up","down","out","over","about","into","through","off",
    "agent","caller","cbre","dispatch","talk","hi","hello","okay","ok","um","uh",
    "yeah","yep","mm","hmm","right","sure","thanks","please","alright","got",
}

_TOKEN_RX = re.compile(r"\b[a-z]{3,}\b")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RX.findall(text.lower()) if t not in STOP_WORDS]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _extract_features(transcripts: list[str]) -> list[set[str]]:
    """Return per-transcript set of (unigrams ∪ bigrams)."""
    out = []
    for t in transcripts:
        toks = _tokenize(t)
        out.append(set(toks) | set(_ngrams(toks, 2)))
    return out


def _lift_keywords(
    positives: list[str],
    negatives: list[str],
    min_lift: float = 4.0,
    min_pos_rate: float = 0.30,
    max_keywords: int = 8,
) -> list[tuple[str, float]]:
    """Find features that appear ≥min_pos_rate in positives and have lift ≥min_lift."""
    pos_feats = _extract_features(positives)
    neg_feats = _extract_features(negatives)

    pos_count: Counter = Counter()
    for s in pos_feats:
        pos_count.update(s)
    neg_count: Counter = Counter()
    for s in neg_feats:
        neg_count.update(s)

    n_pos = max(len(positives), 1)
    n_neg = max(len(negatives), 1)

    scored: list[tuple[str, float, float]] = []
    for kw, c in pos_count.items():
        pos_rate = c / n_pos
        if pos_rate < min_pos_rate:
            continue
        neg_rate = (neg_count.get(kw, 0) + 1) / (n_neg + 1)  # Laplace smoothing
        lift = pos_rate / neg_rate
        if lift >= min_lift:
            scored.append((kw, lift, pos_rate))

    # Prefer specific multi-word phrases when they overlap with their unigrams.
    scored.sort(key=lambda x: (-x[1], -len(x[0]), -x[2]))
    return [(kw, round(lift, 2)) for kw, lift, _ in scored[:max_keywords]]


def _load_data() -> tuple[list[dict], dict, dict]:
    """Return (predictions, labels_by_id, transcripts_by_id)."""
    preds = json.loads(PREDICTIONS.read_text())
    labels = json.loads(DEV_LABELS.read_text())["by_id"]
    transcripts = {
        r["transcript_id"]: r for r in json.loads(DEV_TRANSCRIPTS.read_text())
    }
    return preds, labels, transcripts


def _full_transcript(turns: list[dict]) -> str:
    return " ".join(
        f"[{t['speaker']}] {t['text']}"
        for t in turns
    )


def _run_shadow_eval(traps_json_path: Path, out_path: Path) -> None:
    cmd = [
        sys.executable, str(EVAL_DIR / "run_eval.py"),
        "--agent", "agent:classify",
        "--out", str(out_path),
        "--traps", str(traps_json_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(HERE))


def _score(predictions_path: Path) -> dict:
    sys.path.insert(0, str(EVAL_DIR))
    from scoring import grade, index_preds, load_json  # type: ignore

    eval_rows = load_json(DEV_TRANSCRIPTS)
    gt_raw = load_json(DEV_LABELS)
    gt_map = gt_raw.get("by_id", gt_raw)
    preds = load_json(predictions_path)
    preds_map = index_preds(preds)
    return grade(preds_map, eval_rows, gt_map)


def main(rebuild_baseline: bool = False) -> None:
    MODELS.mkdir(exist_ok=True)

    print("Loading predictions, labels, transcripts ...")
    preds, labels, transcripts = _load_data()

    # 1. Bucket transcripts by (ai_sub, true_sub).
    positives_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    negatives_by_sub: dict[str, list[str]] = defaultdict(list)
    for pred in preds:
        tid = pred["transcript_id"]
        label = labels.get(tid)
        if not label:
            continue
        tlog = pred.get("trainer_log") or {}
        ai = (tlog.get("ai_prediction") or {}).get("subcategory")
        if not ai:
            continue
        true_sub = label.get("true_subcategory")
        tx = transcripts.get(tid, {}).get("turns", [])
        text = _full_transcript(tx)
        if ai in DANGEROUS:
            if ai != true_sub:
                positives_by_pair[(ai, true_sub)].append(text)
            else:
                # Same dangerous subcategory, correctly classified — negative
                # for the differential analysis.
                negatives_by_sub[ai].append(text)

    # 2. For each candidate transition with n_cases >= MIN_CASES, mine keywords.
    MIN_CASES = 3  # small dev set — relax for n
    candidates = []
    for (ai_sub, true_sub), pos_transcripts in positives_by_pair.items():
        if len(pos_transcripts) < MIN_CASES:
            continue
        negs = negatives_by_sub.get(ai_sub, [])
        if not negs:
            continue
        kws = _lift_keywords(pos_transcripts, negs)
        if not kws:
            continue
        candidate = {
            "dangerous_subcategory": ai_sub,
            "safer_subcategory": true_sub,
            "safer_category": SUB_TO_CAT.get(true_sub, "JANITORIAL"),
            "max_risk": SAFER_MAX_RISK.get(true_sub, "MEDIUM"),
            "n_cases": len(pos_transcripts),
            "n_negative_baseline": len(negs),
            "extracted_keywords": [kw for kw, _ in kws],
            "lift_scores": {kw: lift for kw, lift in kws},
            "example_transcripts": pos_transcripts[:3],
        }
        candidates.append(candidate)

    if not candidates:
        REPORT.write_text(json.dumps({
            "n_candidates": 0,
            "message": "No (ai_sub, true_sub) transitions met the MIN_CASES threshold "
                       "or no keywords cleared the lift/rate bar.",
            "candidates": [],
        }, indent=2))
        print(f"\nNo candidates. Wrote empty report → {REPORT}")
        return

    print(f"Found {len(candidates)} candidate(s). Running shadow evals ...")

    # 3. Baseline eval — reuse predictions.json (already produced).
    baseline = _score(PREDICTIONS)
    baseline_composite = baseline["composite"]
    baseline_false_911 = baseline["false_911_count"]
    print(f"  baseline composite={baseline_composite:.2f}  false_911={baseline_false_911}")

    # 4. For each candidate, shadow-eval with that single trap appended.
    for cand in candidates:
        trap = {
            "dangerous": cand["dangerous_subcategory"],
            "keywords": cand["extracted_keywords"],
            "correction": {
                "subcategory": cand["safer_subcategory"],
                "category": cand["safer_category"],
                "max_risk": cand["max_risk"],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([trap], f)
            traps_path = Path(f.name)
        out_path = HERE / f"predictions_shadow_{cand['dangerous_subcategory']}_to_{cand['safer_subcategory']}.json"
        print(f"\n  Shadow eval: {cand['dangerous_subcategory']} → {cand['safer_subcategory']} "
              f"(keywords: {cand['extracted_keywords'][:3]}...)")
        try:
            _run_shadow_eval(traps_path, out_path)
            shadow = _score(out_path)
            cand["dev_composite_baseline"] = round(baseline_composite, 4)
            cand["dev_composite_with_trap"] = round(shadow["composite"], 4)
            cand["composite_delta"] = round(shadow["composite"] - baseline_composite, 4)
            cand["false_911_baseline"] = baseline_false_911
            cand["false_911_with_trap"] = shadow["false_911_count"]
            cand["false_911_delta"] = shadow["false_911_count"] - baseline_false_911
            if cand["composite_delta"] > 0 and cand["false_911_delta"] <= 0:
                cand["recommendation"] = "ADOPT"
            elif cand["false_911_delta"] > 0:
                cand["recommendation"] = "REJECT_SAFETY"
            else:
                cand["recommendation"] = "REVIEW"
            print(f"    composite Δ={cand['composite_delta']:+.4f}  "
                  f"false_911 Δ={cand['false_911_delta']:+d}  "
                  f"→ {cand['recommendation']}")
        except subprocess.CalledProcessError as e:
            cand["recommendation"] = "EVAL_FAILED"
            cand["error"] = str(e)
        finally:
            traps_path.unlink(missing_ok=True)

    REPORT.write_text(json.dumps({
        "n_candidates": len(candidates),
        "baseline_composite": baseline_composite,
        "baseline_false_911": baseline_false_911,
        "candidates": candidates,
    }, indent=2))
    print(f"\nWrote {REPORT}")
    for c in sorted(candidates, key=lambda x: -x.get("composite_delta", 0)):
        print(f"  {c['dangerous_subcategory']:>15} → {c['safer_subcategory']:<20}  "
              f"n={c['n_cases']:>2}  Δ={c.get('composite_delta', 0):+.4f}  "
              f"{c.get('recommendation','?')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-baseline", action="store_true",
                    help="(reserved) — current version reuses predictions.json")
    args = ap.parse_args()
    main(rebuild_baseline=args.rebuild_baseline)
