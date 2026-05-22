"""
Intake nodes — first-contact processing of the caller's transcript.

Maps to the brief's deliverables on:
  - **Inarticulate callers / missing information** — extract_and_check uses
    gpt-4.1-mini to pull structured fields out of free-form caller speech,
    plus a quality-check that emits a follow-up question when the description
    is too vague.
  - **Clarification policy** — clarification_gate runs 4 deterministic
    heuristics (sparse caller, taxonomy echo, floor self-correction, floor
    exceeds building) to detect ambiguous calls before they're classified;
    the classify_issue LLM then reviews the gate's verdict.
  - **Multi-turn / follow-up loop** — gather_followup pauses interactively
    via interrupt() to ask the caller a clarifying question, accumulating
    answers back into the transcript before re-extraction.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from .config import SUBCATEGORY_TO_CATEGORY, _building_by_id
from .prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_TEMPLATE
from .state import ClarificationDecision, EntityWithQuality, WorkflowState
from .utils import _log_cache_usage, _match_building, _normalize_floor


# ── Clarification-gate heuristics ────────────────────────────────────────────

SPARSE_FIRST_TURN_WORDS = 6
SPARSE_TOTAL_WORDS = 35

SUBCATEGORY_SURFACE_FORMS = {
    sub: [sub.replace("_", " ")] for sub in SUBCATEGORY_TO_CATEGORY
}

_GENERIC_OPENING_RX = re.compile(
    r"(?:there'?s an?|i mean,?|you know,?)?\s*(?:issue|problem)\s+with\s+("
    + "|".join(
        re.escape(s)
        for forms in SUBCATEGORY_SURFACE_FORMS.values()
        for s in forms
    )
    + r")",
    re.IGNORECASE,
)
_FLOOR_NUM_RX = re.compile(r"\bfloor\s+(\d+)\b", re.IGNORECASE)


def _sparse_caller(turns: list[dict]) -> Optional[Tuple[int, int]]:
    """Fires when the caller's first utterance is short (<=6 words) AND
    their total speech across all turns is short (<=35 words). Sparse calls
    on safety-sensitive symptoms often warrant follow-up.
    """
    caller = [t.get("text", "") for t in turns if t.get("speaker") == "caller"]
    if not caller:
        return None
    first = len(caller[0].split())
    total = sum(len(t.split()) for t in caller)
    if first <= SPARSE_FIRST_TURN_WORDS and total <= SPARSE_TOTAL_WORDS:
        return (first, total)
    return None


def _generic_opening(turns: list[dict]) -> Optional[str]:
    """Fires when the caller's first utterance literally repeats a taxonomy
    label ('issue with no heating'). Signal that an operator pulled the
    subcategory out of them rather than the caller volunteering a symptom.
    """
    first = next(
        (t.get("text", "") for t in turns if t.get("speaker") == "caller"), ""
    )
    m = _GENERIC_OPENING_RX.search(first.lower())
    return m.group(1) if m else None


def _floor_self_correction(turns: list[dict]) -> Optional[Tuple[int, int]]:
    """Fires when the caller names ≥2 different floor numbers across turns.
    Direct evidence that the location they reported is ambiguous.
    """
    seen: list[int] = []
    for t in turns:
        if t.get("speaker") != "caller":
            continue
        for n in _FLOOR_NUM_RX.findall(t.get("text", "")):
            i = int(n)
            if i not in seen:
                seen.append(i)
    return (seen[0], seen[1]) if len(seen) >= 2 else None


def _floor_exceeds_building(
    building: Optional[dict], extracted_floor: Optional[str]
) -> Optional[Tuple[int, int]]:
    """Fires when the extracted floor number is greater than the building's
    floor_count. Either a true location error or a suite number parsed as a
    floor — either way, worth a human check.
    """
    if not building or not extracted_floor:
        return None
    m = re.search(r"\d+", extracted_floor)
    if not m:
        return None
    claimed = int(m.group())
    cap = building.get("floor_count", 999)
    return (claimed, cap) if claimed > cap else None


def _build_question(
    *,
    floor_correction: Optional[Tuple[int, int]],
    floor_exceeds: Optional[Tuple[int, int]],
    sparse: Optional[Tuple[int, int]],
    generic: Optional[str],
) -> str:
    """Pick the most specific clarifying question for whichever heuristic(s) fired."""
    if floor_correction:
        f1, f2 = floor_correction
        return f"Just to confirm — is the issue on Floor {f1} or Floor {f2}?"
    if floor_exceeds:
        _, cap = floor_exceeds
        return f"The building only has {cap} floors — which floor are you on?"
    if sparse or generic:
        return (
            "Could you tell me a bit more about what's going wrong — for example, "
            "what specifically isn't working and how long it's been like that?"
        )
    return "Could you describe the issue in a bit more detail so I can route it correctly?"


def decide_clarification(
    turns: list[dict],
    building: Optional[dict],
    extracted_floor: Optional[str],
) -> ClarificationDecision:
    """Aggregate the 4 heuristics into a single verdict + audit trail."""
    reasons: list[str] = []

    sparse = _sparse_caller(turns)
    generic = _generic_opening(turns)
    floor_correction = _floor_self_correction(turns)
    floor_exceeds = _floor_exceeds_building(building, extracted_floor)

    if sparse:
        reasons.append(f"sparse_caller:first{sparse[0]}_total{sparse[1]}")
    if generic:
        reasons.append(f"taxonomy_echo:{generic.replace(' ', '_')}")
    if floor_correction:
        reasons.append(
            f"floor_self_correction:{floor_correction[0]}_then_{floor_correction[1]}"
        )
    if floor_exceeds:
        reasons.append(
            f"floor_exceeds_building:claimed_{floor_exceeds[0]}_max_{floor_exceeds[1]}"
        )

    needs = bool(reasons)
    question = (
        _build_question(
            floor_correction=floor_correction,
            floor_exceeds=floor_exceeds,
            sparse=sparse,
            generic=generic,
        )
        if needs
        else None
    )

    return ClarificationDecision(
        needs_clarification=needs,
        question=question,
        reasons=tuple(reasons),
    )


# ── Nodes ────────────────────────────────────────────────────────────────────


def make_extract_and_check(llm: ChatOpenAI):
    """Combined entity extraction + quality check in a single LLM call.

    Skips re-extraction when state.extraction_complete is already True
    (set on previous iteration of the gather_followup loop).
    """
    # include_raw=True returns {"raw": AIMessage, "parsed": EntityWithQuality, ...}
    # so we can both get the structured entity AND surface cache-hit telemetry
    # from raw.response_metadata.
    extractor = llm.with_structured_output(EntityWithQuality, include_raw=True)

    def extract_and_check(state: WorkflowState):
        description = state["transcript"]
        profile = state.get("caller_profile")
        if profile:
            description += (
                f"\n\n[CALLER PROFILE — last-known default]"
                f"\n  Name: {profile['caller_name']}"
                f"\n  Building: {profile['primary_building_name']}"
                f"\n  Floor: {profile['primary_floor']}"
            )

        if state.get("extraction_complete"):
            return {"follow_ups": [], "needs_clarification": False}

        # System message is byte-stable — the per-call caller description is
        # the only dynamic content, isolated in the user message.
        out = extractor.invoke([
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(description=description)},
        ])
        _log_cache_usage("extract_and_check", out.get("raw"))
        parsed = out.get("parsed")
        if parsed is None:
            # parsing failed — fall back to a permissive default so the graph
            # can still continue (downstream nodes tolerate missing fields).
            return {"follow_ups": [], "extraction_complete": True, "needs_clarification": False}
        data = parsed.model_dump(mode="json")

        # Split into entity fields and quality fields
        entity = {
            "problem_category": data["problem_category"],
            "affected_system": data.get("affected_system"),
            "building_name": data.get("building_name"),
            "address": data.get("address"),
            "floor": data.get("floor"),
            "location_detail": data.get("location_detail"),
            "severity": data["severity"],
            "urgency_indicators": data.get("urgency_indicators", []),
            "summary": data["summary"],
        }

        needs_clarification = data.get("needs_clarification", False)
        follow_ups = [data["followup_question"]] if data.get("followup_question") else []
        extraction_complete = data.get("description_specific_enough", True) and data.get("location_ok", True)

        return {
            "entity": entity,
            "follow_ups": follow_ups,
            "extraction_complete": extraction_complete,
            "needs_clarification": needs_clarification,
        }

    return extract_and_check


def gather_followup(state: WorkflowState):
    """Pause for caller input via interrupt(). Bypassed in eval mode by the
    needs_followup conditional edge."""
    question = state["follow_ups"][0]

    answer = interrupt({
        "type": "followup_question",
        "question": question,
        "accumulated_so_far": state["transcript"],
    })

    new_transcript = f"{state['transcript']}\n[CALLER] {answer}"
    return {
        "transcript": new_transcript,
        "follow_ups": state["follow_ups"][1:],
        "extraction_complete": False,
    }


def clarification_gate(state: WorkflowState):
    """Deterministic clarification-signal gate.

    Writes its verdict to state.gate_clarification_verdict (reviewable by
    classify_issue) — does NOT touch state.needs_clarification, so the
    downstream LLM can flip the decision in either direction.

    In interactive mode the question is also appended to state.follow_ups so
    the downstream gather_followup node actually asks it. In eval mode the
    needs_followup conditional skips the gather loop, so the appended question
    is a no-op there.
    """
    entity = state.get("entity") or {}
    building = _match_building(entity.get("building_name"))
    if not building and state.get("caller_profile"):
        building = _building_by_id.get(
            state["caller_profile"].get("primary_building_id")
        )

    decision = decide_clarification(
        state.get("turns_split") or [],
        building,
        _normalize_floor(entity.get("floor")),
    )

    existing_follow_ups = list(state.get("follow_ups") or [])
    if decision.question and decision.question not in existing_follow_ups:
        existing_follow_ups.append(decision.question)

    return {
        "gate_clarification_verdict": decision.needs_clarification,
        "clarification_question": decision.question,
        "clarification_reasons": decision.reasons,
        "follow_ups": existing_follow_ups,
    }
