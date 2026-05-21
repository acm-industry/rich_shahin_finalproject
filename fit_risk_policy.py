"""
Build the per-subcategory historical band-cap table.

For each of the 36 subcategories, compute the empirical distribution of
`final_risk_level` over historical tickets. The cap is the *highest* band
where empirical probability >= --threshold-pct (default 3%). At inference,
the LLM's emitted band is demoted to the cap when it exceeds it.

Output:
    models/risk_policy.json   — caps + full distribution + n_per_subcategory
                                + intake_vs_final_mismatch_rate + subcategory_cost_factor

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

    # Group final_risk_level + intake/final band + cost by subcategory
    by_sub: dict[str, dict] = {}
    for r in records:
        sub = (r.get("final_subcategory") or "").lower()
        band = (r.get("final_risk_level") or "").upper()
        if not sub or band not in BAND_ORD:
            continue
        d = by_sub.setdefault(sub, {
            "bands": Counter(),
            "intake_pairs": [],
            "costs": [],
        })
        d["bands"][band] += 1
        intake = (r.get("intake_risk_level") or "").upper()
        if intake in BAND_ORD:
            d["intake_pairs"].append((intake, band))
        parts = r.get("parts_cost_usd") or 0
        labor = r.get("labor_cost_usd") or 0
        cost = parts + labor
        if cost > 0:
            d["costs"].append(cost)

    caps: dict[str, str] = {}
    distribution: dict[str, dict[str, float]] = {}
    n_per_subcategory: dict[str, int] = {}
    # For Risk = P(error) × Cost(error) — historical aggregations:
    intake_vs_final_mismatch_rate: dict[str, float] = {}
    mean_total_cost_usd: dict[str, float] = {}
    threshold = threshold_pct / 100.0

    for sub, d in sorted(by_sub.items()):
        counts = d["bands"]
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

        # intake_vs_final mismatch — fraction of historical tickets whose intake
        # band did NOT match the final band. Feeds the P(error) blend at inference.
        pairs = d["intake_pairs"]
        intake_vs_final_mismatch_rate[sub] = (
            round(sum(1 for i, f in pairs if i != f) / len(pairs), 4)
            if pairs else 0.0
        )

        # Mean dispatch cost — sum(parts_cost_usd + labor_cost_usd) averaged.
        mean_total_cost_usd[sub] = (
            round(sum(d["costs"]) / len(d["costs"]), 2) if d["costs"] else 0.0
        )

    # Normalize subcategory_cost_factor to [0, 1] across subcategories.
    max_cost = max(mean_total_cost_usd.values()) if mean_total_cost_usd else 0.0
    subcategory_cost_factor: dict[str, float] = (
        {sub: round(c / max_cost, 4) for sub, c in mean_total_cost_usd.items()}
        if max_cost > 0 else {sub: 0.0 for sub in mean_total_cost_usd}
    )

    artifact = {
        "threshold_pct": threshold_pct,
        "caps": caps,
        "distribution": distribution,
        "n_per_subcategory": n_per_subcategory,
        # Fields used by the Risk = P(error) × Cost(error) trigger:
        "intake_vs_final_mismatch_rate": intake_vs_final_mismatch_rate,
        "subcategory_cost_factor": subcategory_cost_factor,
        "mean_total_cost_usd": mean_total_cost_usd,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {ARTIFACT}")

    print("\nCap table + Risk = P(error) × Cost(error) inputs:")
    print(f"  {'subcategory':<22} {'cap':<10} {'n':>5}  {'mismatch':>9}  {'cost_norm':>9}  {'mean_$':>8}")
    for sub in sorted(caps):
        print(f"  {sub:<22} {caps[sub]:<10} {n_per_subcategory[sub]:>5}  "
              f"{intake_vs_final_mismatch_rate[sub]:>9.3f}  "
              f"{subcategory_cost_factor[sub]:>9.3f}  "
              f"${mean_total_cost_usd[sub]:>7.0f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-pct", type=float, default=3.0,
                    help="Minimum %% empirical mass for a band to be allowed (default: 3.0)")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the artifact already exists.")
    args = ap.parse_args()
    raise SystemExit(main(args.threshold_pct, args.force))
