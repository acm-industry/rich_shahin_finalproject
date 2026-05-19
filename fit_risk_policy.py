"""
Build the per-subcategory historical band-cap table.

For each of the 36 subcategories, compute the empirical distribution of
`final_risk_level` over historical tickets. The cap is the *highest* band
where empirical probability >= --threshold-pct (default 5%). At inference,
the LLM's emitted band is demoted to the cap when it exceeds it.

Output:
    models/risk_policy.json   — caps + full distribution + n_per_subcategory

Re-run:
    python fit_risk_policy.py            # skips if artifact exists
    python fit_risk_policy.py --force    # always regenerates
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = HERE / "operational"
MODELS = HERE / "models"
ARTIFACT = MODELS / "risk_policy.json"

BANDS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]
BAND_ORD = {b: i for i, b in enumerate(BANDS)}


def main(threshold_pct: float, force: bool) -> int:
    if ARTIFACT.exists() and not force:
        print(f"{ARTIFACT} exists. Pass --force to regenerate.")
        return 0

    MODELS.mkdir(exist_ok=True)
    records = json.loads((OPS / "historical_records.json").read_text())
    print(f"Loaded {len(records)} historical tickets")

    # Group final_risk_level by subcategory
    by_sub: dict[str, Counter] = {}
    for r in records:
        sub = (r.get("final_subcategory") or "").lower()
        band = (r.get("final_risk_level") or "").upper()
        if not sub or band not in BAND_ORD:
            continue
        by_sub.setdefault(sub, Counter())[band] += 1

    caps: dict[str, str] = {}
    distribution: dict[str, dict[str, float]] = {}
    n_per_subcategory: dict[str, int] = {}
    threshold = threshold_pct / 100.0

    for sub, counts in sorted(by_sub.items()):
        total = sum(counts.values())
        n_per_subcategory[sub] = total
        dist = {b: round(counts.get(b, 0) / total, 4) for b in BANDS}
        distribution[sub] = dist

        # Cap = highest band where P(band) >= threshold
        cap = "LOW"
        for b in BANDS:
            if dist[b] >= threshold:
                cap = b
        caps[sub] = cap

    artifact = {
        "threshold_pct": threshold_pct,
        "caps": caps,
        "distribution": distribution,
        "n_per_subcategory": n_per_subcategory,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {ARTIFACT}")

    print("\nCap table:")
    print(f"  {'subcategory':<22} {'cap':<10} {'n':>6}  distribution (LOW / MED / HIGH / EMER)")
    for sub in sorted(caps):
        d = distribution[sub]
        print(f"  {sub:<22} {caps[sub]:<10} {n_per_subcategory[sub]:>6}  "
              f"{d['LOW']:.2f} / {d['MEDIUM']:.2f} / {d['HIGH']:.2f} / {d['EMERGENCY']:.2f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-pct", type=float, default=5.0,
                    help="Minimum %% empirical mass for a band to be allowed (default: 5.0)")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the artifact already exists.")
    args = ap.parse_args()
    raise SystemExit(main(args.threshold_pct, args.force))
