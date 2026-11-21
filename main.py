from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import preprocessing
except ImportError:
    preprocessing = None

try:
    from inverted_index import (
        load_processed_corpus,
        build_inverted_index,
        verify_inverted_index,
        inspect_top_terms,
    )
except ImportError:
    load_processed_corpus = build_inverted_index = verify_inverted_index = inspect_top_terms = None

try:
    from boolean_search import execute_boolean_query, search_text_query
except ImportError:
    execute_boolean_query = search_text_query = None

try:
    from ranked_search import RankedSearch
except ImportError:
    RankedSearch = None


def resolve_documents_csv() -> Path | None:
    """يبحث عن ملف documents.csv في المسارات المحتملة أو يولّده عبر preprocessing إذا أمكن."""
    candidate_paths = [
        BASE_DIR / "documents.csv",
        BASE_DIR / "dataset" / "documents.csv",
        BASE_DIR / "data" / "processed" / "documents.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return path

    if preprocessing is not None and hasattr(preprocessing, "main"):
        try:
            preprocessing.main()
            for path in candidate_paths:
                if path.exists():
                    return path
        except Exception as exc:
            print(f"[تنبيه] تعذّر توليد documents.csv تلقائياً: {exc}")

    return None


def prepare_system(csv_path: Path | None):
    """يجهّز الـ corpus، الفهرس العكسي، وتقرير التحقق، ومحرك البحث المرتّب."""
    corpus: dict[str, list[str]] = {}
    inverted_index: dict[str, set[str]] = {}
    all_docs: set[str] = set()
    verification_report: dict[str, object] | None = None
    ranked_engine = None

    if csv_path and csv_path.exists():
        if load_processed_corpus and build_inverted_index:
            corpus = load_processed_corpus(csv_path)
            inverted_index = build_inverted_index(corpus)
            all_docs = set(corpus.keys())
            if verify_inverted_index:
                verification_report = verify_inverted_index(corpus, inverted_index)

        if RankedSearch:
            try:
                ranked_engine = RankedSearch(csv_path)
            except Exception as exc:
                print(f"[تنبيه] تعذّر تهيئة RankedSearch: {exc}")

    return corpus, inverted_index, all_docs, verification_report, ranked_engine


def handle_boolean_menu(inverted_index: dict[str, set[str]], all_docs: set[str]) -> None:
    """قائمة البحث البولياني عبر execute_boolean_query أو search_text_query."""
    if not inverted_index:
        print("[تنبيه] الفهرس العكسي غير متوفر حالياً.")
        return
    if not (execute_boolean_query or search_text_query):
        print("[تنبيه] وحدة boolean_search غير متوفرة.")
        return

    print("\n--- البحث البولياني (Boolean Search) ---")
    print("أمثلة استعلام: python AND index | python OR search | NOT search")
    raw_query = input("أدخل الاستعلام البولياني: ").strip()
    if not raw_query:
        print("[تنبيه] الاستعلام فارغ.")
        return

    if search_text_query:
        results = sorted(search_text_query(raw_query, inverted_index, all_docs))
    else:
        parts = raw_query.split()
        if len(parts) >= 2 and parts[0].upper() == "NOT":
            results = sorted(execute_boolean_query("NOT", [parts[1]], inverted_index, all_docs))
        elif len(parts) >= 3 and parts[1].upper() in {"AND", "OR"}:
            op = parts[1].upper()
            terms = [parts[0]] + [p for p in parts[2:] if p.upper() != op]
            results = sorted(execute_boolean_query(op, terms, inverted_index, all_docs))
        else:
            results = sorted(execute_boolean_query("OR", [parts[0]], inverted_index, all_docs))

    print(f"عدد المستندات المطابقة: {len(results)}")
    if results:
        print("المستندات:", ", ".join(results))
    else:
        print("لا توجد مستندات مطابقة.")


def handle_ranked_menu(ranked_engine) -> None:
    """قائمة البحث المرتّب (TF-IDF + Cosine Similarity)."""
    if ranked_engine is None:
        print("[تنبيه] محرك البحث المرتّب (RankedSearch) غير متوفر حالياً.")
        return

    print("\n--- البحث المرتّب (Ranked Search - TF-IDF) ---")
    query = input("أدخل نص الاستعلام: ").strip()
    if not query:
        print("[تنبيه] الاستعلام فارغ.")
        return

    top_k_raw = input("كم عدد النتائج المطلوبة (Top-K، الافتراضي 5)? ").strip()
    try:
        top_k = int(top_k_raw) if top_k_raw else 5
    except ValueError:
        print("[تنبيه] قيمة غير صالحة لـ Top-K، سيتم استخدام 5.")
        top_k = 5

    results = ranked_engine.search(query, top_k=top_k)
    if not results:
        print("لا توجد نتائج مطابقة.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"{rank}. doc_id={item['doc_id']} | title={item['title']} | score={item['score']:.4f}")


def main() -> None:
    print("=== نظام استرجاع المعلومات — نقطة الربط والتشغيل (main.py) ===")
    csv_path = resolve_documents_csv()
    if csv_path:
        print(f"تم تحميل ملف البيانات: {csv_path}")
    else:
        print("[تنبيه] لم يتم العثور على documents.csv بعد.")

    corpus, inverted_index, all_docs, report, ranked_engine = prepare_system(csv_path)
    print(f"عدد المستندات: {len(corpus)} | عدد المصطلحات المفهرسة: {len(inverted_index)}")
    if report:
        print(
            f"حالة التحقق من الفهرس: {'صحيح (Valid)' if report.get('is_valid') else 'يوجد ملاحظات'} "
            f"(missing={report.get('missing_terms_count')}, mismatched={report.get('mismatched_postings_count')})"
        )

    while True:
        print("\nاختر العملية:")
        print("1. بحث بولياني (Boolean Search: AND / OR / NOT)")
        print("2. بحث مرتّب (Ranked Search: TF-IDF + Cosine Similarity)")
        print("3. عرض أكثر المصطلحات ظهوراً (Top Terms by DF)")
        print("4. إعادة تشغيل المعالجة المسبقة وتحديث الفهرس")
        print("0. خروج")

        choice = input("رقم الخيار: ").strip()
        if choice == "1":
            handle_boolean_menu(inverted_index, all_docs)
        elif choice == "2":
            handle_ranked_menu(ranked_engine)
        elif choice == "3":
            if inverted_index and inspect_top_terms:
                print("\n--- أكثر المصطلحات ظهوراً ---")
                for term, df in inspect_top_terms(inverted_index, n=10):
                    print(f"{term}: {df} docs")
            else:
                print("[تنبيه] الفهرس غير متوفر.")
        elif choice == "4":
            if preprocessing and hasattr(preprocessing, "main"):
                preprocessing.main()
                csv_path = resolve_documents_csv()
                corpus, inverted_index, all_docs, report, ranked_engine = prepare_system(csv_path)
                print("تم تحديث البيانات والفهرس بنجاح.")
            else:
                print("[تنبيه] وحدة preprocessing غير متوفرة.")
        elif choice == "0":
            print("تم إنهاء البرنامج.")
            break
        else:
            print("[تنبيه] خيار غير صحيح، حاول مرة أخرى.")


if __name__ == "__main__":
    main()
