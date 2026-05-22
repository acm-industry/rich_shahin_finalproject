"""
Location-resolution + vendor-dispatch nodes.

Maps to the brief's deliverables on:
  - **Location fields** (building_name + address + floor) — resolve_location
    fuses the LLM's emitted building_name, the extractor's entity, and the
    caller_profile's last-known building to produce a canonical building
    record (from `buildings.json`) and a normalized floor string.
  - **Vendor-match accuracy** — dispatch_vendor picks one vendor per the
    brief's cascade (specialty → city → building_type → status/rating/SLA);
    when no vendor qualifies, flags `needs_human_review=True` instead of
    auto-dispatching (per the brief's unroutable-case rule).
  - **Category↔subcategory consistency** — dispatch_vendor also enforces the
    taxonomy-derived category from the LLM's subcategory before vendor lookup.
"""
from __future__ import annotations

from .config import (
    CATEGORY_SUBCATEGORIES,
    SUBCATEGORY_TO_CATEGORY,
    _building_by_id,
)
from .state import WorkflowState
from .utils import _match_building, _normalize_floor, _select_vendor


def resolve_location(state: WorkflowState):
    """Pick the canonical building (from `buildings.json`) and a normalized
    floor string. Falls back to the caller_profile's last-known location
    when the LLM/extractor didn't surface one.
    """
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
    """Vendor selection (specialty → city → building_type → status/rating/SLA).

    Also enforces category↔subcategory consistency from the taxonomy: if the
    LLM's subcategory is in `SUBCATEGORY_TO_CATEGORY`, snap category to match.
    If no vendor qualifies AND no prior HITL flag was set, flags for review
    (per the brief: "When nothing qualifies, escalate (don't dispatch).").
    """
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
