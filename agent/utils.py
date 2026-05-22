"""
Shared utilities used by multiple nodes.

  - Transcript flattening (LLM input prep)
  - Profile lookup by phone
  - Building name → record fuzzy match
  - Vendor selection (specialty → city → building_type → status/rating/SLA)
  - Floor-string normalization
  - Word-boundary regex compiler (used by trap patterns + danger-word check)
  - Node-latency instrumentation (timed_node + print_timing_report)
"""
from __future__ import annotations

import re
import time

from .config import (
    _building_by_name,
    _profile_by_phone,
    _vendors,
)


# ── Transcript + profile helpers ─────────────────────────────────────────────


def _flatten(turns: list[dict]) -> str:
    """Format the turns list into the [SPEAKER] line per-line text the LLMs see."""
    return "\n".join(f"[{t['speaker'].upper()}] {t['text']}" for t in turns)


def _lookup_profile(phone: str | None) -> dict | None:
    if not phone:
        return None
    return _profile_by_phone.get(phone)


def _match_building(name: str | None) -> dict | None:
    """Fuzzy match a building name against the 52-entry registry.

    Order of attempts:
      1. Exact (case-insensitive) match
      2. Substring containment (caller's name is a substring of a registry
         name, or vice versa)
      3. Token-overlap: pick the registry entry sharing the most words with
         the caller's input (requires ≥2 shared words to count)
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in _building_by_name:
        return _building_by_name[key]
    for bname, bdata in _building_by_name.items():
        if key in bname or bname in key:
            return bdata
    key_words = set(key.split())
    best, best_score = None, 0
    for bname, bdata in _building_by_name.items():
        overlap = len(key_words & set(bname.split()))
        if overlap > best_score:
            best, best_score = bdata, overlap
    return best if best_score >= 2 else None


# ── Vendor selection ─────────────────────────────────────────────────────────


def _select_vendor(
    subcategory: str, city: str | None, building_type: str | None
) -> str | None:
    """Vendor-match per brief: specialty → city coverage → building_type
    certification → availability/rating/SLA. Returns None if nothing qualifies
    (caller of this function flags `needs_human_review` in that case).
    """
    candidates = [v for v in _vendors if subcategory in v["specialties"]]
    if not candidates:
        return None
    if city:
        city_match = [v for v in candidates if city in v["coverage_cities"]]
        if city_match:
            candidates = city_match
    if building_type:
        type_match = [
            v for v in candidates if building_type in v["building_types_certified"]
        ]
        if type_match:
            candidates = type_match
    if not candidates:
        return None
    status_order = {"available": 0, "at_capacity": 1, "offline": 2}
    candidates.sort(
        key=lambda v: (
            status_order.get(v.get("status_at_last_check", "offline"), 2),
            -v.get("rating", 0),
            v.get("response_sla_minutes", 999),
        )
    )
    return candidates[0]["vendor_id"]


# ── Floor string normalization ───────────────────────────────────────────────


def _normalize_floor(floor) -> str | None:
    """Coerce caller-reported floor strings to the 'Floor N' form the grader
    expects. Leaves prefixed forms (basement, rooftop, mezzanine, ground,
    loading) untouched.
    """
    if not floor:
        return None
    floor = str(floor).strip()
    if floor.isdigit():
        return f"Floor {floor}"
    if len(floor) >= 2 and floor[0].upper() == "F" and floor[1:].isdigit():
        return f"Floor {floor[1:]}"
    if not any(
        floor.lower().startswith(p)
        for p in ("floor", "basement", "rooftop", "mezzanine", "ground", "loading")
    ):
        nums = re.findall(r"\d+", floor)
        if nums:
            return f"Floor {nums[0]}"
    return floor


# ── Word-boundary keyword regex (used by trap patterns + danger-word check) ──


def _compile_kw_regex(keywords):
    """Word-boundary, case-insensitive, longest-first regex over a keyword list."""
    parts = [re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"(?:\b|^)(?:" + "|".join(parts) + r")(?:\b|$)", re.IGNORECASE)


# Shared danger-word regex used by over_escalation_guard's de-escalation
# detection. Distinct from the trap-pattern keyword regexes (which are
# subcategory-specific and live alongside their TRAP_PATTERNS entries).
_DANGER_WORDS_RE = _compile_kw_regex(
    ["fire", "gas", "trapped", "threat", "chemical", "collapse", "smoke", "entrapment"]
)


# ── Per-node latency instrumentation ─────────────────────────────────────────


_node_timings: dict[str, list[float]] = {}


def timed_node(node_name: str, fn):
    """Wrap a node function so the per-call elapsed time is recorded under
    `_node_timings[node_name]`. Used in `_build_graph` so the timing report
    covers every node uniformly.
    """
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        _node_timings.setdefault(node_name, []).append(elapsed)
        return result
    wrapper.__name__ = fn.__name__ if hasattr(fn, "__name__") else node_name
    return wrapper


def print_timing_report():
    """Print a compact per-node + total latency report. Called by `classify`
    after a run completes (or by any external code that wants a summary).
    """
    if not _node_timings:
        return
    print(f"\n{'='*60}")
    print("  NODE LATENCY REPORT")
    print(f"{'='*60}")
    for name, times in sorted(_node_timings.items()):
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  {name:<30} avg={avg:.3f}s  total={total:.2f}s  calls={len(times)}")
    total_all = sum(sum(t) for t in _node_timings.values())
    print(f"  {'TOTAL':<30} {total_all:.2f}s")
    print(f"{'='*60}")
