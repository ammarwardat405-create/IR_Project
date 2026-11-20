from collections import Counter
import json
from pathlib import Path
import re
import pandas as pd

# ------------------------- Configuration -------------------------

INPUT_FILE = Path("dataset-IR.csv")
OUTPUT_FILE = Path("IR_Project15_Prepared_Dataset.csv")

ID_COLUMN = "doc_id"
TITLE_COLUMN = "title"
TEXT_COLUMN = "text"

DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "might", "more", "most", "must", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}


# ------------------------- Text & Helper Functions -------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def simple_stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 3 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def preprocess_text(text: str, stopwords: set[str] | None = None) -> tuple[list[str], list[str]]:
    sw = DEFAULT_STOPWORDS if stopwords is None else stopwords
    raw_tokens = tokenize(text)
    processed_terms = [simple_stem(tok) for tok in raw_tokens if tok not in sw]
    return raw_tokens, processed_terms


def make_unique_ids(ids: list[str]) -> list[str]:
    counts = Counter()
    result = []
    for doc_id in ids:
        counts[doc_id] += 1
        if counts[doc_id] == 1:
            result.append(doc_id)
        else:
            result.append(f"{doc_id}_{counts[doc_id]}")
    return result


def make_unique_titles(titles: list[str]) -> list[str]:
    counts = Counter()
    result = []
    for title in titles:
        counts[title] += 1
        if counts[title] == 1:
            result.append(title)
        else:
            result.append(f"{title} (Passage {counts[title]})")

    return result


# ------------------------- Main preparation pipeline -------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Put this Python file in the same "
            f"folder as '{INPUT_FILE}', or change INPUT_FILE at the top."
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = [ID_COLUMN, TITLE_COLUMN, TEXT_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Columns found in file: {list(data.columns)}"
        )

    # Keep the required fields and preserve original content.
    data = data[required_columns].copy()

    # Normalize missing values and surrounding/repeated whitespace.
    for column in required_columns:
        data[column] = data[column].fillna("").astype(str).str.strip()

    data[TITLE_COLUMN] = data[TITLE_COLUMN].str.replace(r"\s+", " ", regex=True)
    data[TEXT_COLUMN] = data[TEXT_COLUMN].str.replace(r"\s+", " ", regex=True)
    data[ID_COLUMN] = data[ID_COLUMN].str.replace(r"\s+", "", regex=True)

    rows_before = len(data)

    # Remove exact duplicate records.
    data = data.drop_duplicates(
        subset=[ID_COLUMN, TITLE_COLUMN, TEXT_COLUMN],
        keep="first",
    ).copy()
    duplicates_removed = rows_before - len(data)

    # Remove records missing a usable ID, title, or body text.
    before_missing_filter = len(data)
    data = data[
        (data[ID_COLUMN] != "") &
        (data[TITLE_COLUMN] != "") &
        (data[TEXT_COLUMN] != "")
    ].copy()
    empty_records_removed = before_missing_filter - len(data)

    # Ensure unique IDs.
    data[ID_COLUMN] = make_unique_ids(data[ID_COLUMN].tolist())

    # Preserve original title and create unique display titles.
    data["original_title"] = data[TITLE_COLUMN]
    data[TITLE_COLUMN] = make_unique_titles(data[TITLE_COLUMN].tolist())

    # Process every document.
    all_tokens = []
    all_processed_terms = []

    for document_text in data[TEXT_COLUMN]:
        tokens, processed_terms = preprocess_text(document_text)
        all_tokens.append(tokens)
        all_processed_terms.append(processed_terms)

    # Keep token lists as JSON strings so they remain readable in CSV.
    data["tokens"] = [json.dumps(items, ensure_ascii=False) for items in all_tokens]
    data["processed_tokens"] = [
        json.dumps(items, ensure_ascii=False) for items in all_processed_terms
    ]
    data["processed_text"] = [" ".join(items) for items in all_processed_terms]
    data["token_count"] = [len(items) for items in all_processed_terms]

    # Remove records that contain no searchable terms after preprocessing.
    before_processed_filter = len(data)
    data = data[data["processed_text"].str.strip() != ""].copy()
    empty_after_preprocessing = before_processed_filter - len(data)

    # Organize columns for the next project stages.
    output_columns = [
        ID_COLUMN,
        TITLE_COLUMN,
        "original_title",
        TEXT_COLUMN,
        "tokens",
        "processed_tokens",
        "processed_text",
        "token_count",
    ]
    data = data[output_columns]

    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Show one before/after example for the project demonstration.
    print("\n--- Project 15: Dataset Preparation Report ---")
    print(f"Input records: {rows_before}")
    print(f"Exact duplicate records removed: {duplicates_removed}")
    print(f"Empty/incomplete records removed: {empty_records_removed}")
    print(f"Records empty after preprocessing: {empty_after_preprocessing}")
    print(f"Final documents: {len(data)}")
    print(f"Unique document IDs: {data[ID_COLUMN].nunique()}")
    print(f"Unique display titles: {data[TITLE_COLUMN].nunique()}")
    print(f"Output file: {OUTPUT_FILE.resolve()}")

    if not data.empty:
        example = data.iloc[0]
        print("\n--- Before / After Example ---")
        print("Original text:")
        print(example[TEXT_COLUMN])
        print("\nTokens:")
        print(example["tokens"])
        print("\nFinal processed terms:")
        print(example["processed_text"])


if __name__ == "__main__":
    main()
