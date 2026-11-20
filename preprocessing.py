"""
OCR text preprocessing for Sehath Saathi.

Normalizes OCR output and extracts candidate entities (brand, ingredients,
strength, manufacturer, dosage form) for downstream matching.

Design principle: character-level OCR-error correction is applied ONLY in
alphabetic word contexts (brand / ingredient names). It is deliberately NOT
applied to strength strings, so numeric values such as "500 mg" are preserved
exactly as read.
"""
from __future__ import annotations

import re
from typing import Any

# Common OCR substitutions. Only applied to alphabetic word contexts, never to
# strengths or other numeric fields.
_COMMON_OCR_ERRORS = {
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
}

# Units that may legitimately appear in a strength string.
_STRENGTH_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|µg|ml|ml|iu|units?|tablets?|capsules?)\b",
    re.IGNORECASE,
)

_MANUFACTURER_KEYWORDS = {
    "pharma", "labs", "laboratories", "ltd", "limited",
    "inc", "corp", "company", "pharmaceuticals",
}

_DOSAGE_FORMS = {
    "tablet", "tablets", "capsule", "capsules", "syrup", "injection",
    "cream", "ointment", "drop", "drops", "gel", "suspension", "sachet",
}

# A small, clearly labelled reference list of common active ingredients used
# only to help the preprocessor recognise ingredient-like tokens. This is NOT
# a medicine catalogue and contains no dosage advice.
_COMMON_INGREDIENTS = {
    "paracetamol", "acetaminophen", "amoxicillin", "ibuprofen", "aspirin",
    "atorvastatin", "omeprazole", "metformin", "amlodipine", "cetirizine",
}


def normalize_case_and_whitespace(text: str) -> str:
    """Lowercase and collapse internal whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_irrelevant_punctuation(text: str) -> str:
    """Remove punctuation that is not meaningful for medicine matching.

    Digits, decimal points, percent signs and unit letters are preserved so
    strengths such as "500 mg" remain intact.
    """
    text = re.sub(r"[^\w\s\.\%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_common_ocr_errors(text: str, context: str = "word") -> str:
    """Fix common OCR character errors.

    Parameters
    ----------
    text : str
        Input text.
    context : str
        "word"      -> apply substitutions (brand / ingredient names).
        "strength"  -> return text unchanged to preserve numeric strengths.

    Returns
    -------
    str
        Text with OCR errors corrected (word context) or unchanged
        (strength context).
    """
    if context == "strength":
        return text
    return "".join(_COMMON_OCR_ERRORS.get(ch, ch) for ch in text)


def dedupe_lines(lines: list[str]) -> list[str]:
    """Remove consecutive duplicate OCR lines and blank lines."""
    deduped: list[str] = []
    prev: str | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line != prev:
            deduped.append(line)
            prev = line
    return deduped


def extract_strengths(text: str) -> list[str]:
    """Extract possible strength values like '500 mg', '10 ml'."""
    return [m.group(0) for m in _STRENGTH_RE.finditer(text)]


def extract_entities(text: str) -> dict[str, Any]:
    """Extract candidate entities from preprocessed OCR text.

    Returns a dict with:
      normalized_text, deduped_lines, brand_candidates,
      ingredient_candidates, strength_candidates,
      manufacturer_candidates, dosage_form_candidates, tokens
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    deduped = dedupe_lines(lines)
    joined = " ".join(deduped)
    normalized = normalize_case_and_whitespace(joined)

    strengths = extract_strengths(normalized)

    # Remove punctuation for entity extraction but keep strengths readable.
    no_punct = remove_irrelevant_punctuation(normalized)
    tokens = no_punct.split()

    brand_candidates: list[str] = []
    ingredient_candidates: list[str] = []
    manufacturer_candidates: list[str] = []
    dosage_form_candidates: list[str] = []

    for token in tokens:
        token_clean = token.strip(".,;:")
        if not token_clean:
            continue
        # Every alphabetic token (len >= 3) is a potential brand name.
        # Apply OCR error correction for brand/ingredient contexts.
        if token_clean.isalpha() and len(token_clean) >= 3:
            corrected = fix_common_ocr_errors(token_clean, context="word")
            brand_candidates.append(corrected)
        if token_clean in _DOSAGE_FORMS:
            dosage_form_candidates.append(token_clean)
        # Manufacturer: match if token contains a known keyword (e.g. "testpharma" contains "pharma")
        elif any(kw in token_clean for kw in _MANUFACTURER_KEYWORDS):
            manufacturer_candidates.append(token_clean)
        elif token_clean in _COMMON_INGREDIENTS:
            ingredient_candidates.append(fix_common_ocr_errors(token_clean, context="word"))

    return {
        "normalized_text": normalized,
        "deduped_lines": deduped,
        "brand_candidates": brand_candidates,
        "ingredient_candidates": ingredient_candidates,
        "strength_candidates": strengths,
        "manufacturer_candidates": manufacturer_candidates,
        "dosage_form_candidates": dosage_form_candidates,
        "tokens": tokens,
    }


def preprocess_ocr_text(text: str) -> dict[str, Any]:
    """Convenience wrapper: normalize + extract entities in one call."""
    return extract_entities(text)