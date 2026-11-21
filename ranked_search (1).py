"""TF-IDF ranked passage search for Educational Passage Search."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text


class RankedSearch:
    def __init__(self, csv_path):
        columns = ["doc_id", "title", "processed_text"]
        self.documents = pd.read_csv(
            csv_path, usecols=columns, dtype=str, keep_default_na=False,
            encoding="utf-8-sig",
        )
        # Preserve the exact terms produced by preprocessing, including single digits.
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False
        )
        self.document_vectors = self.vectorizer.fit_transform(
            self.documents["processed_text"]
        )

    def search(self, query, top_k=5):
        """Return dictionaries with doc_id, title and cosine score."""
        if top_k <= 0 or not query.strip():
            return []

        _, terms = preprocess_text(query)
        query_vector = self.vectorizer.transform([" ".join(terms)])
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(query_vector, self.document_vectors)[0]
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        results = []
        for index in ranked[:top_k]:
            score = float(scores[index])
            if score <= 0:
                break
            document = self.documents.iloc[index]
            results.append({
                "doc_id": document["doc_id"],
                "title": document["title"],
                "score": score,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = Path(__file__).resolve().parent / "dataset" / "documents.csv"
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("queries", nargs="+", help="Quote each multi-word query.")
    args = parser.parse_args()

    engine = RankedSearch(args.csv)
    for query in args.queries:
        print(f"\nQuery: {query}")
        results = engine.search(query, args.top_k)
        if not results:
            print("No matching passages.")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['doc_id']} | {result['title']} | {result['score']:.4f}")


if __name__ == "__main__":
    main()
