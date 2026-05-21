"""
CBRE Facilities Dispatch Agent — Human-in-the-Loop RAG Pipeline
Exposes: classify(turns, caller_phone) -> dict

LangGraph pipeline adapted from Week5 mvp.py:
  caller_history → extract_entities → check_quality → [gather_followup loop]
  → build_enriched_query → retrieve → grade_documents → [rewrite loop]
  → classify_issue → consensus_check → over_escalation_guard
  → compute_risk_score → resolve_location → dispatch_vendor
  → validator_gate → [override] → trainer_log → END
"""
import json
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / "Week5" / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="Deserializing unregistered type")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# ── Paths & data loading ────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
OPS = HERE / "operational"
CHROMA_DIR = str(HERE / "chroma_db")

_buildings = json.loads((OPS / "buildings.json").read_text())
_vendors = json.loads((OPS / "vendors.json").read_text())
_profiles = json.loads((OPS / "caller_profiles.json").read_text())

_building_by_name: dict[str, dict] = {b["name"].lower(): b for b in _buildings}
_building_by_id: dict[str, dict] = {b["building_id"]: b for b in _buildings}
_profile_by_phone: dict[str, dict] = {p["phone_number"]: p for p in _profiles}

# ── Taxonomy ────────────────────────────────────────────────────────────────

CATEGORY_SUBCATEGORIES = {
    "PLUMBING": ["pipe_leak", "restroom_fixture", "drainage_backup", "roof_leak"],
    "ELECTRICAL": ["power_outage", "panel_hazard", "lighting", "low_voltage_data"],
    "HVAC": ["no_cooling", "no_heating", "air_quality", "refrigerant", "controls_bms"],
    "ELEVATOR": ["entrapment", "malfunction", "minor_issue"],
    "DOORS_ACCESS": ["door_mechanical", "glass_damage", "access_control", "auto_door"],
    "LIFE_SAFETY": ["gas_chemical", "fire_smoke", "structural", "slip_trip"],
    "SECURITY": ["suspicious_person", "unauthorized_access", "active_threat"],
    "JANITORIAL": ["restroom_supplies", "carpet_floor", "waste_odor"],
    "GROUNDS_EXTERIOR": ["parking_lighting", "pavement_damage", "signage_fencing"],
    "PEST_SPECIALTY": ["infestation", "appliance_kitchen", "landscaping"],
}

SUBCATEGORY_TO_CATEGORY: dict[str, str] = {}
for _cat, _subs in CATEGORY_SUBCATEGORIES.items():
    for _sub in _subs:
        SUBCATEGORY_TO_CATEGORY[_sub] = _cat

ALL_SUBCATEGORIES = set(SUBCATEGORY_TO_CATEGORY.keys())

SUBCATEGORY_TO_VENDOR_TYPE = {
    "pipe_leak": "plumber", "restroom_fixture": "plumber",
    "drainage_backup": "plumber", "roof_leak": "roofer",
    "power_outage": "electrician", "panel_hazard": "electrician",
    "lighting": "facilities", "low_voltage_data": "low_voltage_tech",
    "no_cooling": "hvac", "no_heating": "hvac", "air_quality": "hvac",
    "refrigerant": "hvac", "controls_bms": "hvac",
    "entrapment": "elevator_service", "malfunction": "elevator_service",
    "minor_issue": "elevator_service",
    "door_mechanical": "facilities", "glass_damage": "glazier",
    "access_control": "access_control", "auto_door": "access_control",
    "gas_chemical": "fire_life_safety", "fire_smoke": "fire_life_safety",
    "structural": "fire_life_safety", "slip_trip": "janitorial",
    "suspicious_person": "security", "unauthorized_access": "security",
    "active_threat": "security",
    "restroom_supplies": "janitorial", "carpet_floor": "janitorial",
    "waste_odor": "janitorial",
    "parking_lighting": "electrician",
    "pavement_damage": "exterior_maintenance",
    "signage_fencing": "exterior_maintenance",
    "infestation": "pest_control", "appliance_kitchen": "appliance_tech",
    "landscaping": "grounds",
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _flatten(turns: list[dict]) -> str:
    return "\n".join(f"[{t['speaker'].upper()}] {t['text']}" for t in turns)


def _lookup_profile(phone: str | None) -> dict | None:
    if not phone:
        return None
    return _profile_by_phone.get(phone)


def _match_building(name: str | None) -> dict | None:
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


def _select_vendor(
    subcategory: str, city: str | None, building_type: str | None
) -> str | None:
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


def _normalize_floor(floor) -> str | None:
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


# ── Node timing ─────────────────────────────────────────────────────────────

_node_timings: dict[str, list[float]] = {}


def timed_node(node_name: str, fn):
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        _node_timings.setdefault(node_name, []).append(elapsed)
        return result
    wrapper.__name__ = fn.__name__ if hasattr(fn, "__name__") else node_name
    return wrapper


def print_timing_report():
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


# ── Clarification-signal gate ────────────────────────────────────────────────

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


@dataclass(frozen=True)
class ClarificationDecision:
    needs_clarification: bool
    question: Optional[str]
    reasons: Tuple[str, ...]


def _sparse_caller(turns: list[dict]) -> Optional[Tuple[int, int]]:
    caller = [t.get("text", "") for t in turns if t.get("speaker") == "caller"]
    if not caller:
        return None
    first = len(caller[0].split())
    total = sum(len(t.split()) for t in caller)
    if first <= SPARSE_FIRST_TURN_WORDS and total <= SPARSE_TOTAL_WORDS:
        return (first, total)
    return None


def _generic_opening(turns: list[dict]) -> Optional[str]:
    first = next(
        (t.get("text", "") for t in turns if t.get("speaker") == "caller"), ""
    )
    m = _GENERIC_OPENING_RX.search(first.lower())
    return m.group(1) if m else None


def _floor_self_correction(turns: list[dict]) -> Optional[Tuple[int, int]]:
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


# ── Prompts ─────────────────────────────────────────────────────────────────

_BUILDINGS_BLOCK = "\n".join(
    f"- {b['name']} | {b['address']} | {b['city']}" for b in _buildings
)

_TAXONOMY_BLOCK = """PLUMBING:
  pipe_leak        – water leaking from pipes or supply lines
  restroom_fixture – toilet, faucet, or urinal malfunction (not a clogged drain)
  drainage_backup  – clogged drain, sewer backup, toilet overflow
  roof_leak        – water intrusion from roof or skylight
ELECTRICAL:
  power_outage     – total or partial power loss
  panel_hazard     – sparking outlet, hot panel, electrical burn smell, exposed wiring
  lighting         – interior light flickering, dim, or burned out
  low_voltage_data – network, phone, or structured-cabling issue
HVAC:
  no_cooling       – AC not working, space too hot
  no_heating       – heat not working, space too cold
  air_quality      – odors, fumes, poor ventilation (including burnt-food smell, cleaning-product fumes, new-filter off-gassing)
  refrigerant      – chiller or AC refrigerant leak
  controls_bms     – thermostat or building-management-system issue
ELEVATOR:
  entrapment       – person PHYSICALLY TRAPPED inside elevator, cannot exit
  malfunction      – elevator broken, stuck, or OOS with NO ONE trapped inside
  minor_issue      – slow doors, noises, mis-leveling, cosmetic
DOORS_ACCESS:
  door_mechanical  – physical door broken, stuck, won't close or open
  glass_damage     – broken or cracked glass in door, window, or partition
  access_control   – badge reader, keypad, or electronic-lock malfunction
  auto_door        – automatic / powered door not functioning
LIFE_SAFETY:
  gas_chemical     – CONFIRMED gas leak or hazardous-chemical release (NOT cleaning-product fumes or spoiled food)
  fire_smoke       – CONFIRMED visible fire or smoke (NOT burnt-food smell without visible fire)
  structural       – ceiling collapse, falling debris, structural failure (NOT a single water-damaged tile)
  slip_trip        – wet floor, spill, or trip hazard
SECURITY:
  suspicious_person    – unrecognized or concerning individual on premises
  unauthorized_access  – someone in a restricted area without authorization
  active_threat        – violent, armed, or immediately dangerous person on-site
JANITORIAL:
  restroom_supplies – empty soap, paper towels, etc.
  carpet_floor      – stains or spills on carpet / floor
  waste_odor        – trash overflow, odor complaints (non-chemical, including spoiled food)
GROUNDS_EXTERIOR:
  parking_lighting  – parking-lot or garage light out or broken
  pavement_damage   – pothole, cracked sidewalk, parking-lot damage
  signage_fencing   – damaged signage or broken fence
PEST_SPECIALTY:
  infestation       – rodents, insects, pest problem
  appliance_kitchen – break-room appliance malfunction (microwave, fridge, coffee machine)
  landscaping       – irrigation or landscape maintenance"""

EXTRACT_AND_CHECK_PROMPT = (
    "You are a building maintenance analyst. A caller has described a problem.\n"
    "Extract the key information AND assess quality in a single pass.\n"
    "If a field cannot be determined from the description, leave it as null.\n\n"
    "Severity guidelines:\n"
    "- Critical: Immediate danger to life/safety (confirmed gas leak, confirmed fire, trapped persons, flooding near electrical/IT equipment)\n"
    "- High: Significant disruption or escalation risk (major water leak, broken security, HVAC failure affecting many, power outage)\n"
    "- Medium: Needs attention soon but not an emergency (broken door latch, minor plumbing, non-life-safety HVAC)\n"
    "- Low: Minor/cosmetic (flickering light, carpet stain, empty dispenser, cosmetic repairs, burnt food smell)\n\n"
    "IMPORTANT de-escalation rules:\n"
    "- Burnt food smell (popcorn, toast, microwave) without visible fire → Low severity, NOT Critical\n"
    "- Cleaning-product / floor-wax / new-filter fumes → Low severity\n"
    "- Elevator out of service with doors open or person already exited → Medium, not Critical\n"
    "- Upset visitor who left peacefully → Medium, not Critical\n"
    "- Single water-damaged ceiling tile → Medium, not Critical\n\n"
    "Quality check: set description_specific_enough=true if the problem is a specific symptom.\n"
    "Set needs_clarification=true only if the description is genuinely too vague to classify "
    "or a safety-critical detail is ambiguous or location is conflicting.\n\n"
    "Caller description: {description}"
)

CLASSIFY_PROMPT = """You are a CBRE facilities-dispatch AI. Classify the maintenance call below.

TAXONOMY (category: subcategory — description):
{taxonomy}

RISK LEVELS (about 55% of calls are LOW — use LOW as the default unless clearly worse):
- LOW: routine maintenance, NO safety or operational impact. Can wait until next business day.
  Typical: clogged sink (caller reports as routine), slow drain or stopped drain with NO overflow mentioned even if caller says "full", wet floor or trip hazard where caller has already deployed a cone / mopped / blocked the area (hazard mitigated), single suite's AC or heat not working in mild weather with calm caller, badge reader glitch or door latch issue where caller has an alternative entry or workaround, single visible leak / wet spot in one location with NO spreading language, burnt-out bulb, empty soap/supplies, cosmetic damage, carpet stain, slow elevator doors, landscaping, minor door issue (still functions), appliance not working, pest sighting, signage damage, pothole, single flickering light, thermostat slightly off, minor water drip (contained), trash not collected, network/data cable issue, minor plumbing drip, broken fridge.
  Key: if nobody is in danger and operations continue, it's LOW.
  DEFAULT-TO-LOW HEURISTIC: Before picking MEDIUM or higher, run three checks:
    1. Is anyone in physical danger right now? (No → not HIGH or EMERGENCY)
    2. Will operations be visibly disrupted overnight if this waits until tomorrow morning? (No → LOW, not MEDIUM)
    3. Has the caller already mitigated the immediate hazard (cone deployed, power off, area isolated, alternative entry)? (Yes → drop one band)
  If all three favor LOW, use LOW even if a MEDIUM example seems to match the surface words.
- MEDIUM: localized issue affecting one tenant or area, needs same-day attention, NOT a safety hazard. Requires ACTIVE, ongoing impact in this very moment, not just potential for impact.
  Typical (every example has a qualifier — find the qualifier in the transcript before classifying as MEDIUM):
    - one tenant's AC out AND it's a hot day OR caller mentions medical/sensitive context (medical equipment, server room, distressed person)
    - restroom with water actively pooling on the floor (NOT just slow drain or "full" sink)
    - water stain visibly spreading on a wall or ceiling RIGHT NOW
    - door physically broken AND caller cannot secure their suite (no workaround)
    - toilet currently overflowing onto floor (NOT just clogged)
    - access control failure on a RESTRICTED area (server room, executive floor, secured stairwell)
    - pipe leak the caller describes with words like "spreading", "getting worse", "more water now"
  Key: affects comfort or function for a specific area but NOT safety. Most pipe leaks, power issues affecting a single suite, and broken fixtures are MEDIUM at most.
  DO NOT classify as MEDIUM merely because:
    - A caller called the line. Callers report routine issues all the time — they don't have to be MEDIUM.
    - A function is degraded but stable. Broken latch with workaround, slow drain without overflow, AC slightly off — these are LOW.
    - The issue could become worse if ignored. POTENTIAL impact is not ACTIVE impact. A "leaky-looking" wall is LOW until water is spreading.
    - The transcript mentions a part is "broken" or "not working". Many broken things wait until next business day without harm.
    - It "feels like it should be MEDIUM." Trust the LOW default unless the call has SPECIFIC active-impact language.
- HIGH: meaningful safety, operational, or financial risk affecting multiple tenants or the whole building.
  Typical: power out on ENTIRE floor or building, unauthorized person in RESTRICTED area, sparking electrical panel, elevator stuck WITH A PERSON INSIDE (responsive), broken security door to server room, major water leak spreading to multiple floors.
  Key: must affect safety OR multiple tenants/floors. A single-suite power outage is MEDIUM, not HIGH.
- EMERGENCY: immediate threat to life. Warrants 911.
  Typical: confirmed gas leak (rotten egg smell), visible flames, person physically trapped and unresponsive, active armed threat, structural collapse.
  Key: someone could die or be seriously injured RIGHT NOW.

SCOPE TEST: For MEDIUM, find the specific noun in the transcript that names the IMPACT (water on floor / hot day / sensitive context / restricted area / spreading). If you cannot quote that noun verbatim from the transcript, the call is LOW.

─── OVER-ESCALATION RULES (critical — violating these incurs a penalty) ───
These situations are NOT emergencies. Do NOT classify them as fire_smoke, gas_chemical, entrapment, active_threat, or structural:
• Burnt food smell (popcorn, toast, microwave) without visible fire → air_quality or waste_odor, LOW
• Cleaning-product / floor-wax / new-filter fumes → air_quality, MEDIUM
• Spoiled food odor → waste_odor, LOW
• Elevator out of service with doors open OR person already exited → malfunction or minor_issue, LOW/MEDIUM
• Upset visitor who left or was escorted out peacefully → suspicious_person, MEDIUM
• Single water-damaged ceiling tile, drop-ceiling sagging from leak → roof_leak, MEDIUM
• Scorched outlet with breaker tripped safely → panel_hazard, MEDIUM
• Steam from appliance triggering detector → air_quality, LOW

─── MULTI-TURN CORRECTION ───
If the caller changes their mind or corrects themselves during the call, use the FINAL/corrected information.

─── HUMAN REVIEW (needs_human_review) ───
TRUE when:
1. Risk is HIGH or EMERGENCY
2. The call initially sounds dangerous (fire, gas, trapped person, threat, chemical, collapse) but your final classification is benign — a human MUST verify your de-escalation decision
3. Genuinely ambiguous safety situation
4. No vendor available
FALSE when: routine LOW/MEDIUM with clear, unambiguous classification AND no initial danger signals.

─── CLARIFICATION (needs_clarification) ───
TRUE when:
1. The caller's description is genuinely vague — you cannot confidently pick a specific subcategory
2. A safety-critical detail is missing that would change classification (unclear if real fire vs smell, unclear if person trapped vs elevator just broken, unclear severity of hazard)
3. Location information is conflicting or missing (caller mentions one building but profile says another, or no location at all)
FALSE for most calls (~84% don't need it). Only flag when the missing info would materially change dispatch.

─── CLARIFICATION REVIEW ───
Gate decided needs_clarification={gate_verdict} based on signals: {clarification_reasons}.

Context — what this flag means in practice:
  needs_clarification=True routes the call to a human dispatcher for ONE more confirming question
  before the case is closed. It does NOT block classification, dispatch, or vendor selection. It is
  cheap insurance for cases where we want a human to double-check before final action.

Why the gate fires `sparse_caller`: the caller's total speech across all turns was unusually short
  (under ~35 words). Brief calls on safety-sensitive symptoms still need human verification even
  when the symptom name sounds clear — operators have learned that short calls on rare-but-severe
  events (gas, structural, threats) frequently turn out to need follow-up.

Why the gate fires `taxonomy_echo`: the caller's first utterance literally repeated a taxonomy
  label (e.g. "issue with no heating") rather than describing a symptom. This is a strong tell
  that an operator had to draw the subcategory out of them, which usually means more info is needed.

Why the gate fires `floor_self_correction`: the caller named two different floor numbers across
  their turns. Strong tell that location is ambiguous.

Why the gate fires `floor_exceeds_building`: the extracted floor is numerically higher than the
  building's floor_count. Sometimes a true location conflict, sometimes a false alarm from a suite
  number being parsed as a floor.

Default: confirm the gate's verdict. Only flip when one of these clearly applies.

Flip True → False ONLY if BOTH:
  (a) The call is plainly routine maintenance — restroom supplies, light flicker, broken latch,
      sign damage, slow drain, pothole, carpet stain, minor elevator noise, dripping water with
      no urgency, interior door issues, broken appliance, pest sighting, signage damage.
  (b) NO caller turn contains a safety-sensitive word: fire, smoke, gas, eggs, sulfur, chemical,
      solvent, weapon, threat, sparks, structural, ceiling, falling, hazard, evacuation, trapped,
      loitering, unauthorized, emergency, injured, electrical burning, panel.

Calibration examples (sparse_caller TRUE → False, correct flip):
  - "Restroom out of TP. Women's room on three. Pacific Ridge, Floor 3." → routine supply.
  - "Door won't latch, interior door to the suite. Westpark, Floor 2." → routine mechanical.
  - "Sign is propped up but could fall, not blocking anything." → routine signage.
  - "Elevator doing something weird, sounds awful, been like this two days." → minor_issue.
  - "Something dripping from up top, been pouring for an hour." → roof_leak, no urgency words.

Calibration examples (sparse_caller TRUE → KEEP True, do NOT flip):
  - "Ceiling piece fell. Area's clear." → "ceiling", "fell" are safety words → KEEP.
  - "Smell something weird. Rotten eggs." → gas signal → KEEP.
  - "Lights are off. Emergency lights came on." → "emergency" word + safety implication → KEEP.
  - "Someone loitering outside, still there by the elevators." → security word → KEEP.

Flip False → True only when:
  - The caller named a safety-sensitive symptom AND the gate failed to flag it (gate missed).
  - The named floor exceeds the building's floor count, or two contradictory floors appear.

When uncertain, keep the gate's verdict.

─── EMERGENCY SERVICES (dispatched_emergency_services) ───
TRUE ONLY for: confirmed visible fire, confirmed hazardous gas leak, person physically trapped and unable to exit, active threat involving violence or weapons.
FALSE for ALL other cases — especially burnt-food smells, cleaning fumes, empty/open elevator, upset visitors, water damage.

BUILDINGS IN PORTFOLIO:
{buildings}

SIMILAR PAST TICKETS (from historical records):
{rag_context}

EXTRACTED ENTITIES:
{entities}

TRANSCRIPT:
{transcript}
{profile_context}

Classify this call. Return a JSON object with these fields:
  category, subcategory, risk_level, confidence (0-1), reasoning,
  needs_human_review, needs_clarification, dispatched_emergency_services,
  call_summary (2-3 sentence operator-style narrative), building_name, floor"""


# ── Pydantic models ─────────────────────────────────────────────────────────


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EntityWithQuality(BaseModel):
    """Combined entity extraction + quality check in a single LLM call."""
    problem_category: str = Field(
        description="Specific description of the problem (e.g., 'water leaking from ceiling pipe', "
                    "'elevator stuck between floors') — not vague terms like 'issue' or 'problem'"
    )
    affected_system: Optional[str] = Field(
        default=None,
        description="The exact device, system, or component affected"
    )
    building_name: Optional[str] = Field(
        default=None,
        description="Name of the building mentioned in the transcript"
    )
    address: Optional[str] = Field(
        default=None,
        description="Street address mentioned"
    )
    floor: Optional[str] = Field(
        default=None,
        description="Floor number or level (e.g., '3rd floor', 'Floor 7', 'basement')"
    )
    location_detail: Optional[str] = Field(
        default=None,
        description="Specific location within the building (suite, room, wing, hallway)"
    )
    severity: Severity = Field(
        description="Critical / High / Medium / Low based on safety and operational impact"
    )
    urgency_indicators: list[str] = Field(
        default_factory=list,
        description="Specific phrases indicating urgency or timeline"
    )
    summary: str = Field(description="One-sentence summary of the issue")
    # Quality check fields
    description_specific_enough: bool = Field(
        description="True if the problem describes a specific symptom or failure, not vague"
    )
    location_ok: bool = Field(
        default=True,
        description="True if location info is present and not conflicting"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if description is too vague or location is conflicting/missing"
    )
    followup_question: str = Field(
        default="",
        description="Question to get more specific info. Empty string if ok."
    )


SEVERITY_WEIGHTS = {
    "Critical": 1.0,
    "High": 0.75,
    "Medium": 0.4,
    "Low": 0.15,
}

# Band-name → severity weight (mirrors SEVERITY_WEIGHTS but keyed on risk_level strings).
RISK_LEVEL_SEVERITY = {
    "LOW": 0.15,
    "MEDIUM": 0.4,
    "HIGH": 0.75,
    "EMERGENCY": 1.0,
}

# Multiplier on Cost(error) by building_type — captures that a leak in a hospital
# is costlier-to-misclassify than the same leak in a parking garage. Hand-coded
# (no ML); range [0.7, 1.3] gets normalized to [0, 1] in the Cost formula.
BUILDING_TYPE_COST_MODIFIER = {
    "medical": 1.3,
    "hospital": 1.3,
    "industrial": 1.2,
    "warehouse": 1.1,
    "office": 1.0,
    "retail_office": 1.0,
    "parking": 0.7,
}

# HITL trigger threshold for the Risk = P(error) × Cost(error) score. Additive
# only — never overrides an existing False from upstream triggers.
RISK_HITL_THRESHOLD = 0.30

# Over-escalation trap patterns: (dangerous_subcategory, trigger_keywords) → safe correction
TRAP_PATTERNS = [
    {
        "dangerous": "fire_smoke",
        "keywords": ["burnt", "popcorn", "toast", "microwave", "food", "cooking", "burnt food", "smoke smell"],
        "correction": {"subcategory": "air_quality", "category": "HVAC", "max_risk": "LOW"},
    },
    {
        "dangerous": "gas_chemical",
        "keywords": ["cleaning", "wax", "paint", "filter", "chemical smell", "floor wax", "polish", "solvent"],
        "correction": {"subcategory": "air_quality", "category": "HVAC", "max_risk": "LOW"},
    },
    {
        "dangerous": "gas_chemical",
        "keywords": ["spoiled food", "rotten food", "garbage smell", "food smell", "trash smell", "decomposing food"],
        "exclude_keywords": ["rotten eggs", "gas", "propane", "natural gas", "evacuate", "sick", "headache", "911"],
        "correction": {"subcategory": "waste_odor", "category": "JANITORIAL", "max_risk": "LOW"},
    },
    {
        "dangerous": "entrapment",
        "keywords": ["doors open", "got out", "exited", "empty", "no one", "nobody", "already out", "walked out", "open doors"],
        "correction": {"subcategory": "malfunction", "category": "ELEVATOR", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "active_threat",
        "keywords": ["left", "escorted", "gone", "departed", "peaceful", "walked away", "no longer", "already left"],
        "correction": {"subcategory": "suspicious_person", "category": "SECURITY", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "structural",
        "keywords": ["tile", "ceiling tile", "one tile", "stain", "water spot", "single tile", "sagging tile"],
        "correction": {"subcategory": "roof_leak", "category": "PLUMBING", "max_risk": "MEDIUM"},
    },
]

GENUINE_EMERGENCY_SUBCATEGORIES = {"gas_chemical", "fire_smoke", "entrapment", "active_threat", "structural"}


def _compile_kw_regex(keywords):
    """Word-boundary, case-insensitive, longest-first regex over a keyword list."""
    parts = [re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"(?:\b|^)(?:" + "|".join(parts) + r")(?:\b|$)", re.IGNORECASE)


for _trap in TRAP_PATTERNS:
    _trap["_kw_re"] = _compile_kw_regex(_trap["keywords"])
    if _trap.get("exclude_keywords"):
        _trap["_exc_re"] = _compile_kw_regex(_trap["exclude_keywords"])

_DANGER_WORDS_RE = _compile_kw_regex(
    ["fire", "gas", "trapped", "threat", "chemical", "collapse", "smoke", "entrapment"]
)

# ── Graph state ─────────────────────────────────────────────────────────────


class WorkflowState(TypedDict):
    transcript: str
    turns_split: list[dict]
    caller_phone: Optional[str]
    caller_profile: Optional[dict]
    eval_mode: bool
    entity: Optional[dict]
    follow_ups: list
    extraction_complete: bool
    needs_clarification: bool
    gate_clarification_verdict: bool
    clarification_question: Optional[str]
    clarification_reasons: Tuple[str, ...]
    enriched_query: str
    retrieved_docs: list
    retrieved_context: str
    retrieval_attempts: int
    classification: Optional[dict]
    consensus_agreement: Optional[float]
    risk_score: Optional[float]
    risk_details: Optional[dict]
    building: Optional[dict]
    building_name: Optional[str]
    floor: Optional[str]
    address: Optional[str]
    vendor_id: Optional[str]
    _city: Optional[str]
    _building_type: Optional[str]
    needs_human_review: bool
    dispatched_emergency_services: bool
    call_summary: str
    routing_decision: Optional[str]
    human_override: Optional[dict]
    trainer_log: Optional[dict]
    _grade: Optional[str]


# ── Node factories ──────────────────────────────────────────────────────────


def make_extract_and_check(llm: ChatOpenAI):
    """Combined entity extraction + quality check in a single LLM call."""
    extractor = llm.with_structured_output(EntityWithQuality)

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

        result = extractor.invoke(
            EXTRACT_AND_CHECK_PROMPT.format(description=description)
        )
        data = result.model_dump(mode="json")

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
    """Pause for caller input via interrupt(). Bypassed in eval mode."""
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


def build_enriched_query(state: WorkflowState):
    ent = state["entity"]
    enriched = (
        f"Problem: {ent['problem_category']}."
        + (f" System: {ent['affected_system']}." if ent.get("affected_system") else "")
        + f" Severity: {ent['severity']}."
        + f" {ent['summary']}"
    )
    return {
        "enriched_query": enriched,
        "retrieval_attempts": 0,
    }


def make_retrieve(vector_store: Chroma):
    def retrieve(state: WorkflowState):
        """Retrieve necessary historical data based on enriched query"""
        query = state["enriched_query"]
        docs = vector_store.similarity_search(query, k=5)
        retrieved_docs = []
        context_parts = []
        for doc in docs:
            meta = doc.metadata
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": meta,
            })
            context_parts.append(
                f"[Past ticket] {doc.page_content}"
            )
        return {
            "retrieved_docs": retrieved_docs,
            "retrieved_context": "\n\n".join(context_parts),
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return retrieve



def make_classify_issue(llm: ChatOpenAI):
    def classify_issue(state: WorkflowState):
        """Classify issue based on caller profile and the context of the problem that has been parsed"""
        ent = state.get("entity") or {}
        profile = state.get("caller_profile")

        profile_context = ""
        if profile:
            profile_context = (
                f"\nCALLER PROFILE (last-known default — caller may be elsewhere today):\n"
                f"  Name: {profile['caller_name']}\n"
                f"  Building: {profile['primary_building_name']}\n"
                f"  Address: {profile['primary_address']}, {profile['primary_city']}\n"
                f"  Floor: {profile['primary_floor']}\n"
                f"  Suite: {profile.get('primary_suite') or 'N/A'}\n"
                f"  Active: {profile['active']}"
            )

        entities_str = (
            f"Problem: {ent.get('problem_category', 'unknown')}\n"
            f"System: {ent.get('affected_system', 'unknown')}\n"
            f"Building: {ent.get('building_name', 'unknown')}\n"
            f"Floor: {ent.get('floor', 'unknown')}\n"
            f"Severity: {ent.get('severity', 'unknown')}\n"
            f"Summary: {ent.get('summary', '')}"
        )

        reasons = list(state.get("clarification_reasons") or ())
        gate_verdict = state.get("gate_clarification_verdict", False)

        prompt = CLASSIFY_PROMPT.format(
            taxonomy=_TAXONOMY_BLOCK,
            buildings=_BUILDINGS_BLOCK,
            rag_context=state.get("retrieved_context", "None"),
            entities=entities_str,
            transcript=state["transcript"],
            profile_context=profile_context,
            gate_verdict=gate_verdict,
            clarification_reasons=reasons,
        )

        response = llm.invoke(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.content)
        return {"classification": raw}

    return classify_issue


def consensus_check(state: WorkflowState):
    """Two-axis sanity check against the 5 retrieved historical tickets:

    1. Subcategory vote (legacy): if RAG majority disagrees with LLM subcategory,
       weaken confidence by 30%. Does NOT change the label.
    2. Risk-level vote (Strategy 1): demote-only band correction. If RAG majority
       (>=60%) names a band BELOW the LLM's emission, demote to that band. Uses
       per-call historical evidence rather than the aggregate subcategory cap.

    Demote-only by construction — RAG can never promote past the LLM's emission.
    """
    docs = state.get("retrieved_docs") or []
    classification = state["classification"]
    if not docs:
        return {"classification": classification, "consensus_agreement": None}

    # ── Vote 1: subcategory consensus (existing behavior) ──
    sub_votes = Counter(d["metadata"].get("final_subcategory", "") for d in docs)
    majority_sub, majority_count = sub_votes.most_common(1)[0]
    sub_agreement = majority_count / len(docs)
    if sub_agreement >= 0.6 and majority_sub != classification.get("subcategory"):
        conf = classification.get("confidence", 0.5)
        classification["confidence"] = conf * 0.7
        classification["consensus_note"] = (
            f"RAG majority ({sub_agreement:.0%}) suggests {majority_sub}"
        )

    # ── Vote 2: risk-level consensus (demote-only, same-sub filter) ──
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}
    sub = (classification.get("subcategory") or "").lower()
    same_sub_docs = [
        d for d in docs
        if (d["metadata"].get("final_subcategory") or "").lower() == sub
    ]
    if len(same_sub_docs) >= 3:
        band_votes = Counter()
        for d in same_sub_docs:
            band = (d["metadata"].get("final_risk_level") or "").upper()
            if band in band_ord:
                band_votes[band] += 1
        if band_votes:
            majority_band, band_count = band_votes.most_common(1)[0]
            band_agreement = band_count / sum(band_votes.values())
            llm_band = (classification.get("risk_level") or "MEDIUM").upper()
            if (band_agreement >= 0.6
                    and band_ord.get(majority_band, 3) < band_ord.get(llm_band, 1)):
                classification["original_risk_level_rag"] = llm_band
                classification["risk_level"] = majority_band
                classification["consensus_band_applied"] = (
                    f"RAG-band majority ({band_agreement:.0%}={majority_band}, "
                    f"n={len(same_sub_docs)} same-sub) demoted from {llm_band}"
                )

    return {"classification": classification, "consensus_agreement": sub_agreement}


def over_escalation_guard(state: WorkflowState):
    classification = state["classification"]
    transcript = state["transcript"]
    subcategory = (classification.get("subcategory") or "").lower()

    for trap in TRAP_PATTERNS:
        if subcategory == trap["dangerous"]:
            # Check exclude keywords first — if any match, skip this trap
            if trap.get("_exc_re") and trap["_exc_re"].search(transcript):
                continue
            if trap["_kw_re"].search(transcript):
                # Correct any findings of previously defined keywords by capping their risk level ensuring they don't call 911
                correction = trap["correction"]
                classification["subcategory"] = correction["subcategory"]
                classification["category"] = correction["category"]
                risk = (classification.get("risk_level") or "LOW").upper()
                max_risk = correction["max_risk"]
                risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}
                if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                    classification["risk_level"] = max_risk
                classification["dispatched_emergency_services"] = False
                classification["guard_applied"] = (
                    f"De-escalated from {trap['dangerous']} to {correction['subcategory']}"
                )
                # De-escalation from dangerous category → human must verify
                classification["needs_human_review"] = True
                break

    # If LLM initially classified as dangerous but its own output is benign,
    # flag for human review (covers cases the guard didn't catch)
    if _DANGER_WORDS_RE.search(transcript):
        final_sub = (classification.get("subcategory") or "").lower()
        if final_sub not in GENUINE_EMERGENCY_SUBCATEGORIES:
            classification["needs_human_review"] = True

    # Emergency services should only be dispatched for genuine emergencies
    subcategory = (classification.get("subcategory") or "").lower()
    if classification.get("dispatched_emergency_services"):
        if subcategory not in GENUINE_EMERGENCY_SUBCATEGORIES:
            classification["dispatched_emergency_services"] = False
        elif (classification.get("risk_level") or "").upper() not in ("HIGH", "EMERGENCY"):
            classification["dispatched_emergency_services"] = False

    return {"classification": classification}


def make_risk_policy_cap(policy: dict):
    """Per-subcategory historical band cap (demote-only).

    Each subcategory has a "max plausible band" derived from the empirical
    distribution of final_risk_level in historical_records.json. When the LLM
    emits a band that exceeds the cap, demote to the cap. LLM under-estimates
    pass through unchanged.
    """
    caps = policy["caps"]
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

    def risk_policy_cap(state: WorkflowState):
        classification = state["classification"]
        sub = (classification.get("subcategory") or "").lower()
        cap = caps.get(sub, "EMERGENCY")
        llm = (classification.get("risk_level") or "MEDIUM").upper()

        if band_ord.get(llm, 1) > band_ord.get(cap, 3):
            classification["original_risk_level"] = llm
            classification["risk_level"] = cap
            classification["risk_policy_applied"] = (
                f"capped {llm}→{cap} (P[≥{llm}] < {policy.get('threshold_pct', 5)}% for {sub})"
            )

        return {"classification": classification}

    return risk_policy_cap


def make_compute_risk_score(policy: dict):
    """Risk = P(error) × Cost(error) HITL trigger (Design B, additive).

    P(error) is a weighted blend of three deterministic signals:
      - 0.40 × (1 - llm_confidence)           [LLM's own uncertainty]
      - 0.30 × rag_disagreement_strength      [bands the RAG-consensus demoted by]
      - 0.30 × historical_mismatch_rate[sub]  [P(intake_band != final_band) per sub]

    Cost(error) is a weighted blend of three deterministic factors:
      - 0.50 × severity_weight[risk_level]                       [band severity]
      - 0.30 × subcategory_cost_factor[sub]                      [historical $ cost]
      - 0.20 × normalized_building_modifier[building_type]       [criticality]

    Both clamped to [0, 1]. Risk = P × C. Trigger needs_human_review when
    Risk >= RISK_HITL_THRESHOLD, ADDITIVELY (never overrides existing False).
    All existing hard rules preserved (band HIGH/EMERGENCY → review,
    confidence<0.6 → review).
    """
    mismatch_rates = policy.get("intake_vs_final_mismatch_rate", {})
    cost_factors = policy.get("subcategory_cost_factor", {})
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

    # Normalize the BUILDING_TYPE_COST_MODIFIER constants to [0, 1] range.
    # Min=0.7 (parking), Max=1.3 (medical). office (1.0) lands at 0.5.
    _bm_min = min(BUILDING_TYPE_COST_MODIFIER.values())
    _bm_max = max(BUILDING_TYPE_COST_MODIFIER.values())
    _bm_range = _bm_max - _bm_min if _bm_max > _bm_min else 1.0

    def compute_risk_score(state: WorkflowState):
        classification = state["classification"]

        # ── P(error) components ──
        confidence = classification.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence_uncertainty = max(0.0, min(1.0, 1.0 - confidence))

        # RAG-band disagreement strength = bands the consensus demoted by.
        # consensus_check sets original_risk_level_rag only when it demoted.
        rag_disagreement = 0.0
        original_rag = classification.get("original_risk_level_rag")
        if original_rag:
            current_band = (classification.get("risk_level") or "MEDIUM").upper()
            diff = band_ord.get(original_rag.upper(), 1) - band_ord.get(current_band, 1)
            rag_disagreement = min(1.0, max(0.0, diff / 3.0))

        subcat = (classification.get("subcategory") or "").lower()
        historical_mismatch = mismatch_rates.get(subcat, 0.0)

        #get P(error) from weighted signals
        p_error = (
            0.40 * confidence_uncertainty +
            0.30 * rag_disagreement +
            0.30 * historical_mismatch
        )
        p_error = round(min(1.0, max(0.0, p_error)), 4)

        # ── Cost(error) components ──
        risk_level = (classification.get("risk_level") or "MEDIUM").upper()
        severity_weight = RISK_LEVEL_SEVERITY.get(risk_level, 0.4)

        sub_cost = cost_factors.get(subcat, 0.5)

        building_type = (state.get("_building_type") or "").lower()
        bm_raw = BUILDING_TYPE_COST_MODIFIER.get(building_type, 1.0)
        building_modifier_norm = (bm_raw - _bm_min) / _bm_range  # [0, 1]

        cost_error = (
            0.50 * severity_weight +
            0.30 * sub_cost +
            0.20 * building_modifier_norm
        )
        cost_error = round(min(1.0, max(0.0, cost_error)), 4)

        # ── Risk = P × C ──
        risk = round(p_error * cost_error, 4)

        # ── Preserve existing HITL behavior ──
        needs_human = bool(classification.get("needs_human_review", False))
        needs_clarification = state.get("needs_clarification", False) or bool(
            classification.get("needs_clarification", False)
        )

        # Hard rules (kept as-is — additive with the new Risk trigger):
        if risk_level in ("HIGH", "EMERGENCY"):
            needs_human = True
        if confidence < 0.6:
            needs_human = True

        # NEW: Risk-based additive trigger.
        if risk >= RISK_HITL_THRESHOLD:
            needs_human = True

        # Build a compact reason string for trainer-log auditability.
        reason_parts = []
        if risk >= RISK_HITL_THRESHOLD:
            reason_parts.append(
                f"Risk={risk:.2f} (P_err={p_error:.2f} × Cost={cost_error:.2f})"
            )
        if risk_level in ("HIGH", "EMERGENCY"):
            reason_parts.append(f"band={risk_level}")
        if confidence < 0.6:
            reason_parts.append(f"low conf {confidence:.2f}")

        return {
            "risk_score": risk,
            "risk_details": {
                "p_error": p_error,
                "cost_error": cost_error,
                "p_error_components": {
                    "confidence_uncertainty": round(confidence_uncertainty, 4),
                    "rag_disagreement": round(rag_disagreement, 4),
                    "historical_mismatch": round(historical_mismatch, 4),
                },
                "cost_error_components": {
                    "severity_weight": severity_weight,
                    "sub_cost_factor": sub_cost,
                    "building_modifier_norm": round(building_modifier_norm, 4),
                },
                "threshold": RISK_HITL_THRESHOLD,
                "reason": " | ".join(reason_parts) if reason_parts else "within normal parameters",
            },
            "needs_human_review": needs_human,
            "needs_clarification": needs_clarification,
        }

    return compute_risk_score


def resolve_location(state: WorkflowState):
    classification = state["classification"]
    profile = state.get("caller_profile")

    building_name_raw = classification.get("building_name")
    floor = classification.get("floor")

    # Also check entity extraction for building/floor
    entity = state.get("entity") or {}
    if not building_name_raw:
        building_name_raw = entity.get("building_name")
    if not floor:
        floor = entity.get("floor")

    building = _match_building(building_name_raw)

    if not building and profile:
        building = _building_by_id.get(profile.get("primary_building_id"))
    if not floor and profile:
        floor = profile.get("primary_floor")

    floor = _normalize_floor(floor)

    building_name = building["name"] if building else building_name_raw
    address = building["address"] if building else (
        profile["primary_address"] if profile else None
    )
    city = building["city"] if building else (
        profile["primary_city"] if profile else None
    )
    building_type = building["building_type"] if building else (
        profile.get("primary_building_type") if profile else None
    )

    return {
        "building": building,
        "floor": floor,
        "address": address,
        "building_name": building_name,
        "_city": city,
        "_building_type": building_type,
    }


def dispatch_vendor(state: WorkflowState):
    """Dispatching a task to the correct vendor"""
    classification = state["classification"]
    subcategory = (classification.get("subcategory") or "").lower()

    # Fix category ↔ subcategory consistency
    if subcategory in SUBCATEGORY_TO_CATEGORY:
        classification["category"] = SUBCATEGORY_TO_CATEGORY[subcategory]
    elif (classification.get("category") or "").upper() not in CATEGORY_SUBCATEGORIES:
        classification["category"] = "JANITORIAL"

    city = state.get("_city")
    building_type = state.get("_building_type")

    vendor_id = _select_vendor(subcategory, city, building_type)

    needs_human = state.get("needs_human_review", False)
    if not vendor_id and not needs_human:
        needs_human = True

    return {
        "vendor_id": vendor_id,
        "needs_human_review": needs_human,
        "classification": classification,
    }


def validator_gate(state: WorkflowState):
    """HITL validator gate. In eval mode, auto-approves. In interactive mode, uses interrupt()."""
    if state.get("eval_mode"):
        risk_score = state.get("risk_score", 0.0)
        classification = state["classification"]
        cat = classification.get("category", "")
        sub = classification.get("subcategory", "")
        if state.get("needs_human_review"):
            routing = f"Flagged for review → {cat}/{sub} (risk {risk_score:.2f})"
        else:
            routing = f"Auto-routed → {cat}/{sub} (risk {risk_score:.2f})"
        return {"routing_decision": routing}

    # Interactive mode — interrupt for human review
    if state.get("needs_human_review"):
        decision = interrupt({
            "type": "review_required",
            "reason": state.get("risk_details", {}).get("reason", "elevated risk"),
            "risk_score": state.get("risk_score"),
            "classification": state["classification"],
            "risk_details": state.get("risk_details"),
        })

        if decision.get("override_code"):
            return {
                "human_override": decision,
                "routing_decision": f"Human override → {decision.get('override_code')}",
            }
        else:
            return {"routing_decision": f"Human approved → {state['classification'].get('subcategory')}"}

    classification = state["classification"]
    return {
        "routing_decision": f"Auto-routed → {classification.get('category')}/{classification.get('subcategory')} (risk {state.get('risk_score', 0.0):.2f})",
    }


def override_node(state: WorkflowState):
    """Apply human corrections from validator_gate."""
    human_override = state.get("human_override")
    if not human_override:
        return {}

    classification = state["classification"].copy()
    if human_override.get("category"):
        classification["category"] = human_override["category"]
    if human_override.get("subcategory"):
        classification["subcategory"] = human_override["subcategory"]
    if human_override.get("risk_level"):
        classification["risk_level"] = human_override["risk_level"]
    classification["human_reviewed"] = True

    return {"classification": classification}


def make_trainer_log(state: WorkflowState):
    classification = state["classification"]
    category = (classification.get("category") or "").upper()
    subcategory = (classification.get("subcategory") or "").lower()
    risk_level = (classification.get("risk_level") or "MEDIUM").upper()

    if risk_level not in ("LOW", "MEDIUM", "HIGH", "EMERGENCY"):
        risk_level = "MEDIUM"

    building_name = state.get("building_name") or classification.get("building_name")
    floor = state.get("floor")
    address = state.get("address")
    vendor_id = state.get("vendor_id")
    needs_human = state.get("needs_human_review", False)
    needs_clarification = state.get("needs_clarification", False)
    emergency = bool(classification.get("dispatched_emergency_services", False))

    # Call summary with fallback
    call_summary = (classification.get("call_summary") or "").strip()
    if len(call_summary) < 30:
        call_summary = (
            f"Caller reported {subcategory.replace('_', ' ')} at "
            f"{building_name or 'unknown building'}, {floor or 'unknown floor'}. "
            f"Classified {category}/{subcategory}, risk {risk_level}."
        )

    ai_prediction = {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "building_name": building_name,
        "floor": floor,
        "vendor_id": vendor_id,
        "needs_human_review": needs_human,
        "needs_clarification": needs_clarification,
        "dispatched_emergency_services": emergency,
        "original_risk_level": classification.get("original_risk_level"),
        "risk_policy_applied": classification.get("risk_policy_applied"),
        "original_risk_level_rag": classification.get("original_risk_level_rag"),
        "consensus_band_applied": classification.get("consensus_band_applied"),
        "risk_score": state.get("risk_score"),
        "risk_details": state.get("risk_details"),
        "clarification_reasons": list(state.get("clarification_reasons") or ()),
        "confidence": classification.get("confidence"),
    }

    human_override = None
    if state.get("human_override"):
        human_override = state["human_override"]

    final_decision = ai_prediction.copy()
    if human_override:
        for k, v in human_override.items():
            if k in final_decision and v is not None:
                final_decision[k] = v

    trainer_log = {
        "full_transcript": state["transcript"],
        "ai_prediction": ai_prediction,
        "human_override": human_override,
        "final_decision": final_decision,
    }

    return {
        "trainer_log": trainer_log,
        "call_summary": call_summary,
        "dispatched_emergency_services": emergency,
    }


# ── Conditional edge functions ──────────────────────────────────────────────


def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    """Fired AFTER clarification_gate. In interactive mode, route to gather_followup
    when any pending follow_ups exist (either from extract_and_check's LLM extractor
    or from the clarification_gate's appended question). In eval mode, skip directly
    to build_enriched_query — the gate's verdict still drives the needs_clarification
    flag via the LLM review in classify_issue.
    """
    if state.get("eval_mode"):
        return "build_enriched_query"
    if state.get("extraction_complete"):
        return "build_enriched_query"
    if state.get("follow_ups"):
        return "gather_followup"
    return "build_enriched_query"



def override_route(state: WorkflowState) -> Literal["override", "trainer_log"]:
    if state.get("human_override"):
        return "override"
    return "trainer_log"


# ── Graph builder ───────────────────────────────────────────────────────────


def _build_graph(eval_mode: bool = True):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=3, request_timeout=30)
    llm_mini = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3, request_timeout=20)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="historical_records",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    risk_policy = json.loads((HERE / "models" / "risk_policy.json").read_text())

    workflow = StateGraph(WorkflowState)

    # Add nodes (extract+quality merged, grade/rewrite removed for speed)
    workflow.add_node("extract_and_check", timed_node("extract_and_check", make_extract_and_check(llm_mini)))
    workflow.add_node("gather_followup", gather_followup)
    workflow.add_node("clarification_gate", timed_node("clarification_gate", clarification_gate))
    workflow.add_node("build_enriched_query", timed_node("build_enriched_query", build_enriched_query))
    workflow.add_node("retrieve", timed_node("retrieve", make_retrieve(vector_store)))
    workflow.add_node("classify_issue", timed_node("classify_issue", make_classify_issue(llm)))
    workflow.add_node("consensus_check", timed_node("consensus_check", consensus_check))
    workflow.add_node("over_escalation_guard", timed_node("over_escalation_guard", over_escalation_guard))
    workflow.add_node("risk_policy_cap", timed_node("risk_policy_cap", make_risk_policy_cap(risk_policy)))
    workflow.add_node("compute_risk_score", timed_node("compute_risk_score", make_compute_risk_score(risk_policy)))
    workflow.add_node("resolve_location", timed_node("resolve_location", resolve_location))
    workflow.add_node("dispatch_vendor", timed_node("dispatch_vendor", dispatch_vendor))
    workflow.add_node("validator_gate", timed_node("validator_gate", validator_gate))
    workflow.add_node("override", timed_node("override", override_node))
    workflow.add_node("trainer_log", timed_node("trainer_log", make_trainer_log))

    # Edges
    # Flow: extract_and_check → clarification_gate → (gather_followup loop OR build_enriched_query)
    # Running clarification_gate BEFORE the gather conditional means its appended
    # question is available to gather_followup in interactive mode.
    workflow.add_edge(START, "extract_and_check")
    workflow.add_edge("extract_and_check", "clarification_gate")
    workflow.add_conditional_edges("clarification_gate", needs_followup)
    workflow.add_edge("gather_followup", "extract_and_check")
    workflow.add_edge("build_enriched_query", "retrieve")
    workflow.add_edge("retrieve", "classify_issue")
    workflow.add_edge("classify_issue", "consensus_check")
    workflow.add_edge("consensus_check", "over_escalation_guard")
    workflow.add_edge("over_escalation_guard", "risk_policy_cap")
    workflow.add_edge("risk_policy_cap", "compute_risk_score")
    workflow.add_edge("compute_risk_score", "resolve_location")
    workflow.add_edge("resolve_location", "dispatch_vendor")
    workflow.add_edge("dispatch_vendor", "validator_gate")
    workflow.add_conditional_edges("validator_gate", override_route)
    workflow.add_edge("override", "trainer_log")
    workflow.add_edge("trainer_log", END)

    return workflow.compile()


# ── Entry points ────────────────────────────────────────────────────────────

_eval_graph = None


def _get_eval_graph():
    global _eval_graph
    if _eval_graph is None:
        _eval_graph = _build_graph(eval_mode=True)
    return _eval_graph


def classify(turns: list[dict], caller_phone: str | None = None) -> dict:
    graph = _get_eval_graph()
    transcript = _flatten(turns)
    profile = _lookup_profile(caller_phone)

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
        "risk_score": None,
        "risk_details": None,
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

    return _format_prediction(result)


def _format_prediction(result: dict) -> dict:
    classification = result.get("classification") or {}
    category = (classification.get("category") or "").upper()
    subcategory = (classification.get("subcategory") or "").lower()

    # Ensure category↔subcategory consistency
    if subcategory in SUBCATEGORY_TO_CATEGORY:
        category = SUBCATEGORY_TO_CATEGORY[subcategory]

    risk_level = (classification.get("risk_level") or "MEDIUM").upper()
    if risk_level not in ("LOW", "MEDIUM", "HIGH", "EMERGENCY"):
        risk_level = "MEDIUM"

    return {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "needs_human_review": result.get("needs_human_review", False),
        "needs_clarification": result.get("needs_clarification", False),
        "building_name": result.get("building_name"),
        "address": result.get("address"),
        "floor": result.get("floor"),
        "dispatched_vendor_id": result.get("vendor_id"),
        "dispatched_emergency_services": result.get("dispatched_emergency_services", False),
        "call_summary": result.get("call_summary", ""),
        "trainer_log": result.get("trainer_log"),
    }
