"""Mine intake_subcategory → final_subcategory relabel patterns.

Reads operational/historical_records.json. For each intake subcategory, records
the final subcategories it gets re-labeled to and how often. Keeps pairs where
the re-label rate is ≥ MIN_RATE of all intakes for that subcategory, up to
TOP_N_PER_INTAKE targets per intake.

Writes models/intake_drift.json — consumed at module load by agent.py to inject
a hint into CLASSIFY_PROMPT.

Usage:
    python build_drift.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_RATE = 0.05
TOP_N_PER_INTAKE = 3


def main() -> None:
    records = json.loads((ROOT / "operational/historical_records.json").read_text())

    pair_counts: Counter = Counter()
    intake_totals: Counter = Counter()
    for r in records:
        intake = r.get("intake_subcategory")
        final = r.get("final_subcategory")
        if not intake or not final:
            continue
        intake_totals[intake] += 1
        if intake != final:
            pair_counts[(intake, final)] += 1

    drift: dict[str, list[dict]] = defaultdict(list)
    for (i, f), n in pair_counts.items():
        rate = n / intake_totals[i]
        if rate >= MIN_RATE:
            drift[i].append({"to": f, "rate": round(rate, 3), "n": n})

    out = {
        i: sorted(items, key=lambda x: -x["rate"])[:TOP_N_PER_INTAKE]
        for i, items in drift.items()
    }

    (ROOT / "models").mkdir(exist_ok=True)
    dest = ROOT / "models/intake_drift.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dest} — {len(out)} intake subcategories with drift ≥ {MIN_RATE:.0%}")
    for intake, targets in sorted(out.items(), key=lambda kv: -max(t["rate"] for t in kv[1])):
        line = ", ".join(f"{t['to']} ({int(t['rate']*100)}%)" for t in targets)
        print(f"  {intake:<22} → {line}")


if __name__ == "__main__":
    main()
