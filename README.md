# Educational Passage Search Engine — Phase 1

An Information Retrieval (IR) prototype built for **Information Retrieval (902333)** at **Al al-Bayt University** (supervised by Dr. Manar Mizher). The system indexes educational passages derived from **SQuAD** and supports both **Boolean Retrieval** and **Ranked Retrieval (TF-IDF + Cosine Similarity)**.

## Project Structure

```text
.
├── dataset/                # Processed educational passages dataset
├── dataset-IR.csv          # Raw educational passages dataset (500 unique documents)
├── preprocessing.py        # Member 1: Cleaning, tokenization, stopword removal, stemming
├── inverted_index.py       # Member 2: Inverted index construction & verification
├── boolean_search.py       # Member 3: Boolean retrieval (AND, OR, NOT)
├── ranked_search.py        # Member 4: TF-IDF weighting & Cosine Similarity ranking
├── main.py                 # Member 5: Integrated CLI runner & end-to-end orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Team Responsibilities (Phase 1)

* **Member 1 (`preprocessing.py`):** Prepares the 500-document SQuAD passage dataset (`doc_id`, `title`, `text`), removes duplicates/empty rows, normalizes whitespace, applies lowercasing, tokenization, stopword removal, and stemming, and outputs processed tokens and statistics.
* **Member 2 (`inverted_index.py`):** Builds an inverted index mapping each normalized term to a set of matching `doc_id` values, provides posting inspection helpers, and verifies index consistency against the corpus.
* **Member 3 (`boolean_search.py`):** Implements `AND` (intersection), `OR` (union), and `NOT` (complement) queries over the inverted index while handling empty or invalid queries safely.
* **Member 4 (`ranked_search.py`):** Computes TF-IDF weights for documents and queries, calculates Cosine Similarity, and ranks the top-$K$ relevant passages with similarity scores and text snippets.
* **Member 5 (`main.py`, `requirements.txt`, `README.md`):** Integrates all modules into a cohesive command-line interface, documents setup and usage, and prepares Phase 1 deliverables.

## Setup & Execution

1. Ensure you are using **Python 3.10+**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing step if needed:
   ```bash
   python preprocessing.py
   ```
4. Launch the interactive search engine:
   ```bash
   python main.py
   ```
