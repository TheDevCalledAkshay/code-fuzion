import preprocessing

test_texts = [
    "Dolo-650 Paracetamol Tablets IP 650 mg",
    "Dolo-650 Paracetamol Tablets IP Paracetamol IP 650 mg",
    "Dolo-650 Paracetamol Tablets IP Paracetamol IP 850 mg",
]

for text in test_texts:
    entities = preprocessing.preprocess_ocr_text(text)
    print(f"Input: {text}")
    print(f"  brand_candidates: {entities['brand_candidates']}")
    print(f"  ingredient_candidates: {entities['ingredient_candidates']}")
    print(f"  strength_candidates: {entities['strength_candidates']}")
    print(f"  deduped_lines: {entities['deduped_lines']}")
    print()