from pathlib import Path
#from inverted_index import load_processed_corpus, build_inverted_index#


def boolean_and(postings_list: list[set[str]]) -> set[str]:
    """إرجاع المستندات المشتركة بين كل المصطلحات (Intersection)."""
    if not postings_list:
        return set()
    result = set(postings_list[0])
    for postings in postings_list[1:]:
        result &= set(postings)
    return result


def boolean_or(postings_list: list[set[str]]) -> set[str]:
    """إرجاع المستندات التي تحتوي على أي مصطلح (Union)."""
    result: set[str] = set()
    for postings in postings_list:
        result |= set(postings)
    return result


def boolean_not(term_postings: set[str], all_docs: set[str]) -> set[str]:
    """إرجاع المستندات غير الموجودة في مجموعة المصطلح (Complement)."""
    return set(all_docs) - set(term_postings)


def execute_boolean_query(
    operator: str,
    terms: list[str],
    inverted_index: dict[str, set[str]],
    all_docs: set[str],
) -> set[str]:
    """تنفيذ استعلام منطقي (AND, OR, NOT) مع معالجة الاستعلامات الفارغة بأمان."""
    if not terms or not operator:
        return set()

    op = operator.strip().upper()
    clean_terms = [t.strip() for t in terms if t and t.strip()]
    if not clean_terms:
        return set()

    if op == "AND":
        postings_list = [inverted_index.get(term, set()) for term in clean_terms]
        return boolean_and(postings_list)

    if op == "OR":
        postings_list = [inverted_index.get(term, set()) for term in clean_terms]
        return boolean_or(postings_list)

    if op == "NOT":
        term_postings = inverted_index.get(clean_terms[0], set())
        return boolean_not(term_postings, all_docs)

    return set()


def search_text_query(
    query: str,
    inverted_index: dict[str, set[str]],
    all_docs: set[str],
) -> set[str]:
    """
    دعم استعلام نصي بسيط مثل:
    - "NOT term"
    - "term1 AND term2"
    - "term1 OR term2"
    """
    if not query or not query.strip():
        return set()

    parts = query.strip().split()
    if len(parts) == 2 and parts[0].upper() == "NOT":
        return execute_boolean_query("NOT", [parts[1]], inverted_index, all_docs)

    if len(parts) >= 3 and parts[1].upper() in {"AND", "OR"}:
        op = parts[1].upper()
        terms = [parts[0]] + [p for p in parts[2:] if p.upper() != op]
        return execute_boolean_query(op, terms, inverted_index, all_docs)

    return execute_boolean_query("OR", [parts[0]], inverted_index, all_docs)


if __name__ == "__main__":
    sample_index = {
        "python": {"0", "1", "3"},
        "search": {"1", "2"},
        "index": {"0", "2", "4"},
    }
    sample_docs = {"0", "1", "2", "3", "4"}

    print("AND:", sorted(execute_boolean_query("AND", ["python", "index"], sample_index, sample_docs)))
    print("OR:", sorted(execute_boolean_query("OR", ["python", "search"], sample_index, sample_docs)))
    print("NOT:", sorted(execute_boolean_query("NOT", ["search"], sample_index, sample_docs)))
    print("Empty:", sorted(execute_boolean_query("AND", [], sample_index, sample_docs)))