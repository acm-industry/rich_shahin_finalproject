"""
Static configuration + operational data for the CBRE dispatch agent.

This module loads the operational JSON files (buildings, vendors, caller
profiles) once at import time and exposes them alongside the hierarchical
taxonomy (10 categories × 36 subcategories per the brief) and the policy
constants used by the risk-band, dispatch, and HITL layers.

Importing this module also triggers `load_dotenv()` so any module that
constructs an LLM client sees `OPENAI_API_KEY` from the user's .env.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

# .env is at the repo root, one level above this package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = _REPO_ROOT
OPS = HERE / "operational"
MODELS = HERE / "models"
CHROMA_DIR = str(HERE / "chroma_db")

# ── Operational data (loaded once) ────────────────────────────────────────────

_buildings = json.loads((OPS / "buildings.json").read_text())
_vendors = json.loads((OPS / "vendors.json").read_text())
_profiles = json.loads((OPS / "caller_profiles.json").read_text())

# Lookup indices used across nodes
_building_by_name: dict[str, dict] = {b["name"].lower(): b for b in _buildings}
_building_by_id: dict[str, dict] = {b["building_id"]: b for b in _buildings}
_profile_by_phone: dict[str, dict] = {p["phone_number"]: p for p in _profiles}

# ── Hierarchical taxonomy (10 categories × 36 subcategories per the brief) ───

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

SUBCATEGORY_TO_CATEGORY: dict[str, str] = {
    sub: cat
    for cat, subs in CATEGORY_SUBCATEGORIES.items()
    for sub in subs
}

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

# ── Risk-policy constants (Risk = P(error) × Cost(error) HITL trigger) ───────

# Band-name → severity weight, keyed on risk_level strings.
# Feeds the severity term of Cost(error) in compute_risk_score.
RISK_LEVEL_SEVERITY = {
    "LOW": 0.15,
    "MEDIUM": 0.4,
    "HIGH": 0.75,
    "EMERGENCY": 1.0,
}

# Multiplier on Cost(error) by building_type — captures that a leak in a
# hospital is costlier-to-misroute than the same leak in a parking garage.
# Hand-coded (no ML); raw range [0.7, 1.3] gets normalized to [0, 1] in the
# Cost formula at inference time.
BUILDING_TYPE_COST_MODIFIER = {
    "medical": 1.3,
    "hospital": 1.3,
    "industrial": 1.2,
    "warehouse": 1.1,
    "office": 1.0,
    "retail_office": 1.0,
    "parking": 0.7,
}

# HITL trigger threshold for the Risk = P(error) × Cost(error) score.
# Additive only — never overrides an existing False from upstream triggers.
RISK_HITL_THRESHOLD = 0.30

# Subcategories where a confirmed-EMERGENCY truly warrants 911 dispatch.
# Used by `over_escalation_guard` to clamp `dispatched_emergency_services`
# when the LLM mistakenly tries to dispatch on a benign subcategory.
GENUINE_EMERGENCY_SUBCATEGORIES = {
    "gas_chemical", "fire_smoke", "entrapment", "active_threat", "structural"
}
