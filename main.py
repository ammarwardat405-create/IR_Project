from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set

from boolean_search import execute_boolean_query, search_text_query
from ranked_search import (
    build_tfidf_index,
    load_documents_for_ranking,
    ranked_search,
)

try:
    from preprocessing import preprocess_text
except ImportError:
    def preprocess_text(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

try:
    from inverted_index import (
        build_inverted_index,
        get_postings,
        term_document_frequency,
        verify_inverted_index,
    )
except ImportError:
    def build_inverted_index(corpus: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        index: Dict[str, Set[str]] = {}
        for doc_id, terms in corpus.items():
            for term in terms:
                index.setdefault(term, set()).add(doc_id)
        return index

    def get_postings(index: Dict[str, Set[str]], term: str) -> List[str]:
        return sorted(index.get(term.lower().strip(), set()))

    def term_document_frequency(index: Dict[str, Set[str]], term: str) -> int:
        return len(index.get(term.lower().strip(), set()))

    def verify_inverted_index(
        corpus: Dict[str, List[str]], index: Dict[str, Set[str]]
    ) -> bool:
        return build_inverted_index(corpus) == index


BOOL_OPERATORS = {"AND", "OR", "NOT", "(", ")"}


def normalize_boolean_query(query: str) -> str:
    """
    تحويل الكلمات العادية في الاستعلام المنطقي لتتوافق مع المصطلحات المعالجة
    مع الإبقاء على عوامل AND / OR / NOT كما هي.
    """
    tokens = re.findall(r"\(|\)|\w+", query)
    normalized_parts: List[str] = []

    for tok in tokens:
        upper = tok.upper()
        if upper in BOOL_OPERATORS:
            normalized_parts.append(upper)
        else:
            processed = preprocess_text(tok)
            if processed:
                normalized_parts.append(processed[0])
            else:
                normalized_parts.append(tok.lower())

    return " ".join(normalized_parts)


def resolve_dataset_path() -> Path:
    """البحث عن ملف المستندات المعالجة في المسارات المحتملة."""
    candidates = [
        Path("documents.csv"),
        Path("dataset/documents.csv"),
        Path("dataset/IR_Project15_Prepared_Dataset.csv"),
        Path("documents (4).csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    # البحث عن أي ملف CSV معالج داخل مجلد dataset
    dataset_dir = Path("dataset")
    if dataset_dir.exists() and dataset_dir.is_dir():
        csv_files = sorted(dataset_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]
    raise FileNotFoundError(
        "لم يتم العثور على ملف البيانات المعالجة (documents.csv أو داخل مجلد dataset)."
    )


def load_processed_terms_map(filepath: Path) -> Dict[str, List[str]]:
    """استخراج خريطة {doc_id: [processed_tokens]} لبناء الفهرس العكسي."""
    terms_map: Dict[str, List[str]] = {}
    with filepath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            doc_id = str(row.get("doc_id") or f"D{idx}").strip()
            raw_tokens = str(row.get("processed_tokens") or "").strip()
            if raw_tokens.startswith("[") and raw_tokens.endswith("]"):
                try:
                    tokens = json.loads(raw_tokens)
                except json.JSONDecodeError:
                    tokens = ast.literal_eval(raw_tokens)
            elif row.get("processed_text"):
                tokens = str(row["processed_text"]).split()
            else:
                tokens = str(row.get("tokens") or "").split()

            terms_map[doc_id] = [str(t).strip() for t in tokens if str(t).strip()]
    return terms_map


def print_boolean_results(
    doc_ids: Set[str], documents: Dict[str, dict], limit: int = 10
) -> None:
    """طباعة نتائج البحث المنطقي مع العناوين."""
    sorted_ids = sorted(doc_ids)
    print(f"عدد المستندات المطابقة: {len(sorted_ids)}")
    if not sorted_ids:
        print("لا توجد مستندات مطابقة.")
        return

    for doc_id in sorted_ids[:limit]:
        title = documents.get(doc_id, {}).get("title", "Unknown")
        print(f"  - [{doc_id}] {title}")

    if len(sorted_ids) > limit:
        print(f"  ... وتم إخفاء {len(sorted_ids) - limit} نتيجة إضافية.")


def print_ranked_results(results: List[dict]) -> None:
    """طباعة نتائج البحث المرتب TF-IDF."""
    if not results:
        print("لا توجد مستندات مطابقة أو الاستعلام فارغ.")
        return

    for item in results:
        print(
            f"  #{item['rank']} [{item['doc_id']}] {item['title']} "
            f"| Score: {item['score']}\n"
            f"     Snippet: {item['snippet']}"
        )


def run_cli() -> None:
    """تشغيل الواجهة التفاعلية لمحرك البحث التعليمي (Phase 1)."""
    dataset_path = resolve_dataset_path()
    documents = load_documents_for_ranking(dataset_path)
    terms_map = load_processed_terms_map(dataset_path)
    inv_index = build_inverted_index(terms_map)
    all_doc_ids = set(documents.keys())
    doc_vectors, idf_scores = build_tfidf_index(documents)

    is_valid = verify_inverted_index(terms_map, inv_index)
    print("=" * 60)
    print("Educational Passage Search Engine - Phase 1 (Al al-Bayt University)")
    print(f"Dataset Loaded: {dataset_path} | Documents: {len(documents)} | Unique Terms: {len(inv_index)}")
    print(f"Inverted Index Integrity Check: {'PASSED' if is_valid else 'WARNING'}")
    print("=" * 60)

    while True:
        print("\nاختر العملية:")
        print("1) Boolean Search (AND / OR / NOT)")
        print("2) Ranked Search (TF-IDF + Cosine Similarity)")
        print("3) Inspect Term Postings & DF")
        print("4) Exit")
        choice = input("رقم الخيار: ").strip()

        if choice == "1":
            print("أمثلة استعلامات: party AND government | antarctica OR guam | NOT tourism")
            raw_query = input("أدخل الاستعلام المنطقي: ").strip()
            prepared_query = normalize_boolean_query(raw_query)
            matched = search_text_query(prepared_query, inv_index, all_doc_ids)
            print_boolean_results(matched, documents)

        elif choice == "2":
            query = input("أدخل استعلام البحث المرتب: ").strip()
            top_k_raw = input("عدد النتائج المطلوبة (افتراضي 5): ").strip()
            top_k = int(top_k_raw) if top_k_raw.isdigit() and int(top_k_raw) > 0 else 5
            results = ranked_search(query, documents, doc_vectors, idf_scores, top_k=top_k)
            print_ranked_results(results)

        elif choice == "3":
            raw_term = input("أدخل الكلمة (مثل party أو parti): ").strip()
            processed_list = preprocess_text(raw_term)
            term = processed_list[0] if processed_list else raw_term.lower()
            df = term_document_frequency(inv_index, term)
            postings = get_postings(inv_index, term)
            print(f"المصطلح الفعلي في الفهرس: '{term}' | Document Frequency (DF): {df}")
            print(f"أول 15 مستنداً: {postings[:15]}")

        elif choice == "4":
            print("تم إنهاء البرنامج بنجاح.")
            break

        else:
            print("خيار غير صالح، حاول مرة أخرى.")


if __name__ == "__main__":
    run_cli()
