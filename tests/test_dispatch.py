"""Unit tests for vendor selection and location resolution (no API calls)."""
import pytest

from agent import utils
from agent.config import ALL_SUBCATEGORIES, _vendors
from agent.dispatch import dispatch_vendor, resolve_location
from agent.utils import _match_building, _normalize_floor, _select_vendor


def _vendor(vid, *, specialties=("pipe_leak",), cities=("Los Angeles",), types=("office",),
            status="available", rating=4.0, sla=120):
    return {
        "vendor_id": vid,
        "specialties": list(specialties),
        "coverage_cities": list(cities),
        "building_types_certified": list(types),
        "status_at_last_check": status,
        "rating": rating,
        "response_sla_minutes": sla,
    }


@pytest.fixture
def roster(monkeypatch):
    """Replace the real vendor roster with a small controlled one."""
    def _install(*vendors):
        monkeypatch.setattr(utils, "_vendors", list(vendors))
    return _install


# ── _select_vendor: specialty → city → building type → status/rating/SLA ─────


def test_no_vendor_with_the_specialty_returns_none(roster):
    roster(_vendor("v1", specialties=["lighting"]))
    assert _select_vendor("pipe_leak", "Los Angeles", "office") is None


def test_specialty_is_a_hard_filter_even_for_a_better_rated_vendor(roster):
    roster(
        _vendor("wrong_trade", specialties=["lighting"], rating=5.0),
        _vendor("plumber", specialties=["pipe_leak"], rating=3.0),
    )
    assert _select_vendor("pipe_leak", "Los Angeles", "office") == "plumber"


def test_in_city_vendor_beats_better_rated_out_of_city_vendor(roster):
    roster(
        _vendor("far", cities=["San Diego"], rating=5.0),
        _vendor("near", cities=["Los Angeles"], rating=3.5),
    )
    assert _select_vendor("pipe_leak", "Los Angeles", "office") == "near"


def test_city_filter_degrades_gracefully_when_nobody_covers_the_city(roster):
    roster(_vendor("far", cities=["San Diego"]))
    assert _select_vendor("pipe_leak", "Los Angeles", "office") == "far"


def test_building_type_certification_narrows_within_city(roster):
    roster(
        _vendor("office_only", types=["office"], rating=5.0),
        _vendor("medical_cert", types=["medical"], rating=3.5),
    )
    assert _select_vendor("pipe_leak", "Los Angeles", "medical") == "medical_cert"


def test_available_beats_at_capacity_beats_offline(roster):
    roster(
        _vendor("offline", status="offline", rating=5.0),
        _vendor("busy", status="at_capacity", rating=4.9),
        _vendor("free", status="available", rating=3.0),
    )
    assert _select_vendor("pipe_leak", "Los Angeles", "office") == "free"


def test_rating_breaks_ties_then_faster_sla(roster):
    roster(
        _vendor("slow", rating=4.5, sla=240),
        _vendor("fast", rating=4.5, sla=60),
        _vendor("low_rated", rating=4.0, sla=30),
    )
    assert _select_vendor("pipe_leak", "Los Angeles", "office") == "fast"


def test_missing_city_and_type_still_selects_by_specialty(roster):
    roster(_vendor("v1"))
    assert _select_vendor("pipe_leak", None, None) == "v1"


def test_real_roster_only_lists_known_subcategories():
    """Data check: a typo in vendors.json would silently make a vendor unroutable."""
    for v in _vendors:
        assert set(v["specialties"]) <= ALL_SUBCATEGORIES, v["vendor_id"]


# ── dispatch_vendor node ─────────────────────────────────────────────────────


def _dispatch(subcategory, category="PLUMBING", needs_human_review=False):
    return dispatch_vendor({
        "classification": {"subcategory": subcategory, "category": category},
        "_city": "Los Angeles",
        "_building_type": "office",
        "needs_human_review": needs_human_review,
    })


def test_dispatch_fails_closed_when_no_vendor_qualifies(roster):
    roster()
    out = _dispatch("pipe_leak")
    assert out["vendor_id"] is None
    assert out["needs_human_review"] is True


def test_dispatch_does_not_flag_review_when_a_vendor_is_found(roster):
    roster(_vendor("v1"))
    out = _dispatch("pipe_leak")
    assert out["vendor_id"] == "v1"
    assert out["needs_human_review"] is False


def test_dispatch_keeps_an_existing_review_flag(roster):
    roster(_vendor("v1"))
    assert _dispatch("pipe_leak", needs_human_review=True)["needs_human_review"] is True


def test_dispatch_snaps_category_to_match_subcategory(roster):
    roster(_vendor("v1", specialties=["fire_smoke"]))
    out = _dispatch("fire_smoke", category="PLUMBING")
    assert out["classification"]["category"] == "LIFE_SAFETY"


# ── Location helpers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("7", "Floor 7"),
    (7, "Floor 7"),
    ("F12", "Floor 12"),
    ("7th floor", "Floor 7"),
    ("Floor 3", "Floor 3"),
    ("Basement", "Basement"),
    ("Rooftop", "Rooftop"),
    ("", None),
    (None, None),
])
def test_normalize_floor(raw, expected):
    assert _normalize_floor(raw) == expected


def test_match_building_exact_substring_and_token_overlap():
    exact = _match_building("westfield commerce center")
    assert exact["building_id"] == "bld_001"
    assert _match_building("Westfield Commerce")["building_id"] == "bld_001"   # substring
    assert _match_building("the Commerce Center at Westfield")["building_id"] == "bld_001"  # tokens


def test_match_building_rejects_unknown_names():
    assert _match_building("Hogwarts Castle") is None
    assert _match_building(None) is None


def test_resolve_location_falls_back_to_caller_profile():
    profile = {
        "primary_building_id": "bld_001",
        "primary_floor": "Floor 5",
        "primary_address": "100 Wilshire Blvd",
        "primary_city": "Los Angeles",
        "primary_building_type": "office",
    }
    out = resolve_location({"classification": {}, "entity": {}, "caller_profile": profile})
    assert out["building_name"] == "Westfield Commerce Center"
    assert out["address"] == "100 Wilshire Blvd"
    assert out["floor"] == "Floor 5"
    assert out["_city"] == "Los Angeles"


def test_resolve_location_prefers_what_the_caller_said_over_the_profile():
    profile = {"primary_building_id": "bld_001", "primary_floor": "Floor 5",
               "primary_address": "100 Wilshire Blvd", "primary_city": "Los Angeles"}
    other = next(b for b in utils._building_by_name.values() if b["building_id"] != "bld_001")
    out = resolve_location({
        "classification": {"building_name": other["name"], "floor": "9"},
        "entity": {},
        "caller_profile": profile,
    })
    assert out["building_name"] == other["name"]
    assert out["address"] == other["address"]
    assert out["floor"] == "Floor 9"
