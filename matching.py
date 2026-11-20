"""
Medicine matching engine for Sehath Saathi.

Compares preprocessed OCR entities against database records using a
transparent, weighted scoring system. Returns ranked candidate matches
with details on which attributes matched, were missing, or were
contradictory.

IMPORTANT: A matching score is a heuristic ranking score, NOT a medically
validated probability. It should never be used to make medical decisions.
"""
from __future__ import annotations

import difflib
from typing import Any

# Weights must sum to 100. Each attribute contributes weight * similarity / 100.
WEIGHTS = {
    "brand_name": 40,
    "active_ingredients": 30,
    "strength": 15,
    "manufacturer": 10,
    "dosage_form": 5,
}

# Classification thresholds on the 0-100 total score.
GOOD_MATCH_THRESHOLD = 75.0
NO_MATCH_THRESHOLD = 40.0
# If the top two candidates are within this many points, the result is ambiguous.
AMBIGUITY_GAP_POINTS = 2.0

# Similarity band boundaries for classifying a single attribute.
STRONG_MATCH = 0.75   # attribute clearly matches
WEAK_MATCH = 0.40     # below this, attribute is treated as missing


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between 0 and 1 using SequenceMatcher."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _similarity_boosted(a: str, b: str) -> float:
    """Similarity with a boost when one string contains the other.

    Helps partial matches like OCR "amoxicillin" vs DB "amoxicillin trihydrate".
    """
    base = _similarity(a, b)
    a_lower = a.lower()
    b_lower = b.lower()
    if a_lower and b_lower and (a_lower in b_lower or b_lower in a_lower):
        return max(base, 0.85)
    return base


def _best_match(queries: list[str], candidates: list[str]) -> tuple[str, float]:
    """Best similarity score across all query-candidate pairs."""
    best_score = 0.0
    best_val = ""
    for q in queries:
        for c in candidates:
            score = _similarity_boosted(q, c)
            if score > best_score:
                best_score = score
                best_val = c
    return best_val, best_score


def _classify_attribute(score: float) -> str:
    """Classify a single attribute similarity as matched / missing / contradictory."""
    if score >= STRONG_MATCH:
        return "matched"
    if score < WEAK_MATCH:
        return "missing"
    return "contradictory"


def match_medicine(
    entities: dict[str, Any],
    medicines: list[dict],
) -> dict[str, Any]:
    """Match OCR-extracted entities against a list of medicine records.

    Parameters
    ----------
    entities : dict
        Output of preprocessing.extract_entities / preprocess_ocr_text.
    medicines : list[dict]
        Medicine records from the database (each with id, brand_name,
        active_ingredients, strength, dosage_form, manufacturer, aliases).

    Returns
    -------
    dict with keys:
        overall_status : "match" | "ambiguous" | "no_match"
        score : float (0-100) - score of the top candidate
        candidates : list[dict] - ranked candidate details
    """
    brand_queries = entities.get("brand_candidates", [])
    ingredient_queries = entities.get("ingredient_candidates", [])
    strength_queries = entities.get("strength_candidates", [])
    manufacturer_queries = entities.get("manufacturer_candidates", [])
    dosage_form_queries = entities.get("dosage_form_candidates", [])

    results: list[dict[str, Any]] = []

    for med in medicines:
        score_breakdown: dict[str, dict[str, Any]] = {}
        matched_attrs: list[str] = []
        missing_attrs: list[str] = []
        contradictory_attrs: list[str] = []

        # ----- Brand name (includes aliases) -----
        db_brand = med.get("brand_name", "") or ""
        db_aliases = [
            a.strip() for a in (med.get("aliases") or "").split(",") if a.strip()
        ]
        brand_options = [db_brand] + db_aliases
        best_brand, brand_score = _best_match(brand_queries, brand_options)
        score_breakdown["brand_name"] = {
            "weight": WEIGHTS["brand_name"],
            "raw_score": round(brand_score, 4),
            "weighted_score": round(brand_score * WEIGHTS["brand_name"], 4),
            "matched_value": best_brand,
            "db_value": db_brand,
        }
        classification = _classify_attribute(brand_score)
        if classification == "matched":
            matched_attrs.append("brand_name")
        elif classification == "missing":
            missing_attrs.append("brand_name")
        else:
            contradictory_attrs.append("brand_name")

        # ----- Active ingredients -----
        db_ingredients = med.get("active_ingredients", "") or ""
        best_ing, ing_score = _best_match(ingredient_queries, [db_ingredients])
        score_breakdown["active_ingredients"] = {
            "weight": WEIGHTS["active_ingredients"],
            "raw_score": round(ing_score, 4),
            "weighted_score": round(ing_score * WEIGHTS["active_ingredients"], 4),
            "matched_value": best_ing,
            "db_value": db_ingredients,
        }
        classification = _classify_attribute(ing_score)
        if classification == "matched":
            matched_attrs.append("active_ingredients")
        elif classification == "missing":
            missing_attrs.append("active_ingredients")
        else:
            contradictory_attrs.append("active_ingredients")

        # ----- Strength -----
        db_strength = med.get("strength", "") or ""
        best_str, str_score = _best_match(strength_queries, [db_strength])
        score_breakdown["strength"] = {
            "weight": WEIGHTS["strength"],
            "raw_score": round(str_score, 4),
            "weighted_score": round(str_score * WEIGHTS["strength"], 4),
            "matched_value": best_str,
            "db_value": db_strength,
        }
        classification = _classify_attribute(str_score)
        if classification == "matched":
            matched_attrs.append("strength")
        elif classification == "missing":
            missing_attrs.append("strength")
        else:
            contradictory_attrs.append("strength")

        # ----- Manufacturer -----
        db_manufacturer = med.get("manufacturer", "") or ""
        best_mfr, mfr_score = _best_match(manufacturer_queries, [db_manufacturer])
        score_breakdown["manufacturer"] = {
            "weight": WEIGHTS["manufacturer"],
            "raw_score": round(mfr_score, 4),
            "weighted_score": round(mfr_score * WEIGHTS["manufacturer"], 4),
            "matched_value": best_mfr,
            "db_value": db_manufacturer,
        }
        classification = _classify_attribute(mfr_score)
        if classification == "matched":
            matched_attrs.append("manufacturer")
        elif classification == "missing":
            missing_attrs.append("manufacturer")
        else:
            contradictory_attrs.append("manufacturer")

        # ----- Dosage form -----
        db_dosage = med.get("dosage_form", "") or ""
        best_dos, dos_score = _best_match(dosage_form_queries, [db_dosage])
        score_breakdown["dosage_form"] = {
            "weight": WEIGHTS["dosage_form"],
            "raw_score": round(dos_score, 4),
            "weighted_score": round(dos_score * WEIGHTS["dosage_form"], 4),
            "matched_value": best_dos,
            "db_value": db_dosage,
        }
        classification = _classify_attribute(dos_score)
        if classification == "matched":
            matched_attrs.append("dosage_form")
        elif classification == "missing":
            missing_attrs.append("dosage_form")
        else:
            contradictory_attrs.append("dosage_form")

        total_score = sum(v["weighted_score"] for v in score_breakdown.values())

        results.append(
            {
                "medicine_id": med.get("id"),
                "brand_name": db_brand,
                "score": round(total_score, 2),
                "score_breakdown": score_breakdown,
                "matched_attributes": matched_attrs,
                "missing_attributes": missing_attrs,
                "contradictory_attributes": contradictory_attrs,
            }
        )

    # Rank by score descending.
    results.sort(key=lambda x: x["score"], reverse=True)

    # ----- Determine overall status -----
    if not results:
        return {"overall_status": "no_match", "score": 0.0, "candidates": []}

    top = results[0]

    # No match if the top score is below threshold.
    if top["score"] < NO_MATCH_THRESHOLD:
        return {
            "overall_status": "no_match",
            "score": top["score"],
            "candidates": results,
        }

    # Ambiguous if the top two candidates are within the gap.
    if len(results) >= 2:
        gap = top["score"] - results[1]["score"]
        if gap <= AMBIGUITY_GAP_POINTS:
            return {
                "overall_status": "ambiguous",
                "score": top["score"],
                "candidates": results,
            }

    return {
        "overall_status": "match",
        "score": top["score"],
        "candidates": results,
    }