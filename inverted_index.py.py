from pathlib import Path
import csv
import json


def load_processed_corpus(filepath: str | Path) -> dict[str, list[str]]:
    """
    يقرأ النسخة المعدّلة من مجموعة البيانات (CSV أو JSON) ويُرجع قاموساً:
    {doc_id: [term1, term2, ...]}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ملف البيانات المعدّلة غير موجود في المسار: {path}")

    corpus: dict[str, list[str]] = {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # يدعم شكل قائمة من الكائنات [{"doc_id": "1", "tokens": [...]}] أو قاموس {"1": [...]}
        if isinstance(raw_data, list):
            for idx, item in enumerate(raw_data):
                doc_id = str(item.get("doc_id", idx))
                tokens_raw = item.get("tokens") or item.get("terms") or item.get("text") or []
                if isinstance(tokens_raw, str):
                    tokens = tokens_raw.split()
                else:
                    tokens = [str(t) for t in tokens_raw if str(t).strip()]
                corpus[doc_id] = tokens
        elif isinstance(raw_data, dict):
            for doc_id, tokens_raw in raw_data.items():
                if isinstance(tokens_raw, str):
                    tokens = tokens_raw.split()
                else:
                    tokens = [str(t) for t in tokens_raw if str(t).strip()]
                corpus[str(doc_id)] = tokens

    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                doc_id = str(row.get("doc_id") or row.get("id") or idx)
                content = row.get("tokens") or row.get("terms") or row.get("processed_text") or row.get("text") or ""
                # إذا كانت مصفوفة نصية مثل "['a', 'b']" أو نص مفصول بمسافات
                content_str = str(content).strip()
                if content_str.startswith("[") and content_str.endswith("]"):
                    parsed = json.loads(content_str.replace("'", '"'))
                    tokens = [str(t) for t in parsed if str(t).strip()]
                else:
                    tokens = [t for t in content_str.split() if t.strip()]
                corpus[doc_id] = tokens
    else:
        raise ValueError("صيغة الملف غير مدعومة حالياً. استخدم .csv أو .json")

    return corpus


def build_inverted_index(corpus: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    يبني الفهرس العكسي باستخدام dict و set:
    {term: {doc_id1, doc_id2, ...}}
    """
    inverted_index: dict[str, set[str]] = {}

    for doc_id, terms in corpus.items():
        for term in terms:
            clean_term = term.strip()
            if not clean_term:
                continue
            if clean_term not in inverted_index:
                inverted_index[clean_term] = set()
            inverted_index[clean_term].add(doc_id)

    return inverted_index


# --- أدوات فحص القوائم (Postings Inspection Utilities) ---

def get_postings(inverted_index: dict[str, set[str]], term: str) -> set[str]:
    """يُرجع مجموعة معرّفات المستندات التي يظهر فيها المصطلح."""
    return inverted_index.get(term.strip(), set())


def term_document_frequency(inverted_index: dict[str, set[str]], term: str) -> int:
    """يُرجع عدد المستندات التي يظهر فيها المصطلح (DF)."""
    return len(get_postings(inverted_index, term))


def inspect_top_terms(inverted_index: dict[str, set[str]], n: int = 10) -> list[tuple[str, int]]:
    """يعرض أكثر n مصطلحات ظهوراً من حيث عدد المستندات."""
    ranked = sorted(
        ((term, len(docs)) for term, docs in inverted_index.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:n]


# --- التحقق من المصطلحات المفهرسة مقابل الـ Corpus ---

def verify_inverted_index(
    corpus: dict[str, list[str]],
    inverted_index: dict[str, set[str]],
) -> dict[str, object]:
    """
    يتحقق من تطابق الفهرس العكسي مع الـ corpus:
    1. هل كل مصطلح في الـ corpus موجود في الفهرس؟
    2. هل يوجد مصطلح في الفهرس غير موجود في الـ corpus؟
    3. هل قائمة المستندات لكل مصطلح تطابق ظهوره الفعلي في الـ corpus؟
    """
    expected_postings: dict[str, set[str]] = {}
    for doc_id, terms in corpus.items():
        for term in terms:
            clean_term = term.strip()
            if clean_term:
                expected_postings.setdefault(clean_term, set()).add(doc_id)

    expected_terms = set(expected_postings.keys())
    indexed_terms = set(inverted_index.keys())

    missing_terms = expected_terms - indexed_terms
    extra_terms = indexed_terms - expected_terms
    mismatched_postings = {
        term: {
            "expected_docs": sorted(expected_postings.get(term, set())),
            "actual_docs": sorted(inverted_index.get(term, set())),
        }
        for term in (expected_terms & indexed_terms)
        if expected_postings[term] != inverted_index[term]
    }

    is_valid = (not missing_terms) and (not extra_terms) and (not mismatched_postings)

    return {
        "is_valid": is_valid,
        "total_docs": len(corpus),
        "total_indexed_terms": len(indexed_terms),
        "missing_terms_count": len(missing_terms),
        "extra_terms_count": len(extra_terms),
        "mismatched_postings_count": len(mismatched_postings),
        "missing_terms_sample": sorted(missing_terms)[:10],
        "extra_terms_sample": sorted(extra_terms)[:10],
        "mismatched_postings_sample": dict(list(mismatched_postings.items())[:5]),
    }


if __name__ == "__main__":
    # عدّل اسم الملف حسب النسخة المعدّلة الموجودة داخل data/processed/
    processed_file = Path("documents.csv")

    corpus_data = load_processed_corpus(processed_file)
    index = build_inverted_index(corpus_data)
    report = verify_inverted_index(corpus_data, index)

    print("=== Verification Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n=== Top Terms by Document Frequency ===")
    for term, df in inspect_top_terms(index, n=10):
        print(f"{term}: {df} docs")