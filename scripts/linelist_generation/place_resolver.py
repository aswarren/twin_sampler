# place_resolver.py
from geopy.geocoders import Nominatim
import pycountry
from rapidfuzz import process, fuzz
import unidecode
import time

GEOCODER_USER_AGENT = "place_resolver_example"
_geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=10)

# Optional manual overrides for common or ambiguous names
OVERRIDES = {
    "virginia, united states": {"country": "USA", "state": "VA"},
    "england, united kingdom": {"country": "GBR", "state": "ENG"},
}

def _normalize(s):
    return unidecode.unidecode(s or "").strip().lower()

def _country_alpha3_from_name(name):
    if not name:
        return None
    name_norm = _normalize(name)
    # try exact matches
    country = pycountry.countries.get(name=name)
    if not country:
        country = pycountry.countries.get(common_name=name)
    if not country:
        # fuzzy match on country names
        choices = [c.name for c in pycountry.countries]
        match, score, idx = process.extractOne(name_norm, choices, scorer=fuzz.token_sort_ratio)
        if score >= 85:
            country = pycountry.countries.get(name=match)
    return country.alpha_3 if country else None

def _subdivision_code_from_admin(admin_name, country_alpha2):
    if not admin_name or not country_alpha2:
        return None
    # try exact subdivision match
    subs = list(pycountry.subdivisions.get(country_code=country_alpha2) or [])
    if not subs:
        return None
    admin_norm = _normalize(admin_name)
    # exact name match
    for s in subs:
        if _normalize(s.name) == admin_norm:
            # return the short part after the dash if present (e.g., 'US-VA' -> 'VA')
            return s.code.split("-", 1)[1] if "-" in s.code else s.code
    # fuzzy match
    names = [s.name for s in subs]
    match, score, idx = process.extractOne(admin_norm, names, scorer=fuzz.token_sort_ratio)
    if score >= 85:
        s = subs[idx]
        return s.code.split("-", 1)[1] if "-" in s.code else s.code
    # try common abbreviations in the subdivision object (if available)
    for s in subs:
        if hasattr(s, "type") and _normalize(s.type) == admin_norm:
            return s.code.split("-", 1)[1] if "-" in s.code else s.code
    return None

def _country_alpha2_from_alpha3(alpha3):
    if not alpha3:
        return None
    try:
        c = pycountry.countries.get(alpha_3=alpha3)
        return c.alpha_2
    except Exception:
        return None

def fallback_initials(s):
    parts = [p for p in _normalize(s).replace("-", " ").split() if p and p not in ("of","the","province","state","region")]
    if not parts:
        return None
    initials = "".join(p[0].upper() for p in parts)[:4]
    return initials

def resolve_place(text, use_overrides=True, geocode_delay=1.0):
    """
    Input: free text like "Virginia, United States" or "Charlottesville, VA, USA"
    Output: dict like {"country":"USA", "state":"VA"} or best-effort partial result
    """
    if not text or not text.strip():
        return {}

    key = _normalize(text)
    if use_overrides and key in OVERRIDES:
        return OVERRIDES[key].copy()

    # 1) Geocode to get structured address
    try:
        loc = _geolocator.geocode(text, addressdetails=True)
        # be polite to Nominatim
        time.sleep(geocode_delay)
    except Exception:
        loc = None

    country_alpha3 = None
    state_code = None

    if loc and "address" in loc.raw:
        addr = loc.raw["address"]
        # prefer country code and admin fields from geocoder
        country_code = addr.get("country_code", "").upper()  # alpha-2
        country_name = addr.get("country")
        admin = addr.get("state") or addr.get("province") or addr.get("region") or addr.get("county")
        # map to alpha-3
        if country_code:
            try:
                country = pycountry.countries.get(alpha_2=country_code)
                country_alpha3 = country.alpha_3
            except Exception:
                country_alpha3 = _country_alpha3_from_name(country_name)
        else:
            country_alpha3 = _country_alpha3_from_name(country_name)
        # subdivision code via pycountry subdivisions
        if country_alpha3:
            country_alpha2 = _country_alpha2_from_alpha3(country_alpha3)
            state_code = _subdivision_code_from_admin(admin, country_alpha2)
    else:
        # no geocode result: try to parse text heuristically
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 2:
            # assume last part is country
            country_alpha3 = _country_alpha3_from_name(parts[-1])
            if country_alpha3:
                country_alpha2 = _country_alpha2_from_alpha3(country_alpha3)
                state_code = _subdivision_code_from_admin(parts[-2], country_alpha2)
        else:
            # single token: try country first
            country_alpha3 = _country_alpha3_from_name(parts[0])
            if not country_alpha3:
                # maybe it's a state name only; try US as default
                country_alpha3 = "USA"
                state_code = _subdivision_code_from_admin(parts[0], "US")

    # final fallbacks
    if not country_alpha3:
        # try fuzzy country match on whole text
        country_alpha3 = _country_alpha3_from_name(text)

    if not state_code:
        # try to extract state-like token from text
        tokens = [t for t in [p.strip() for p in text.split(",")] if t]
        for t in tokens:
            # skip if t looks like a country
            if _country_alpha3_from_name(t):
                continue
            # try US subdivisions if country unknown
            state_code = _subdivision_code_from_admin(t, "US")
            if state_code:
                country_alpha3 = country_alpha3 or "USA"
                break

    # if still missing, produce initials fallback
    if not state_code:
        # try last non-country token
        tokens = [t for t in [p.strip() for p in text.split(",")] if t]
        candidate = tokens[-2] if len(tokens) >= 2 else tokens[0]
        state_code = fallback_initials(candidate)

    result = {}
    if country_alpha3:
        result["country"] = country_alpha3
    if state_code:
        result["state"] = state_code
    return result

# Quick test
if __name__ == "__main__":
    tests = [
        "Virginia, United States",
        "VA, USA",
        "Charlottesville, Virginia, United States",
        "Ontario, Canada",
        "Bavaria, Germany",
        "New South Wales, Australia",
        "Île-de-France, France",
    ]
    for t in tests:
        print(t, "->", resolve_place(t))

