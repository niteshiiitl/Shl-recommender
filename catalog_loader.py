"""
Loads and indexes the SHL product catalog for retrieval.
Builds a simple TF-IDF + keyword index over all Individual Test Solutions.
"""
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Any

CATALOG_PATH = Path(__file__).parent / "catalog.json"

# Map verbose key names to short codes used in the API response
KEY_CODE_MAP = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def _short_codes(keys: List[str]) -> str:
    codes = [KEY_CODE_MAP.get(k, k[0]) for k in keys]
    # deduplicate while preserving order
    seen = set()
    result = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return ",".join(result)


def load_catalog() -> List[Dict[str, Any]]:
    with open(CATALOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    raw = json.loads(content, strict=False)

    items = []
    for entry in raw:
        if entry.get("status") != "ok":
            continue
        item = {
            "id": entry.get("entity_id", ""),
            "name": entry.get("name", ""),
            "url": entry.get("link", ""),
            "description": entry.get("description", ""),
            "keys": entry.get("keys", []),
            "test_type": _short_codes(entry.get("keys", [])),
            "duration": entry.get("duration", ""),
            "languages": entry.get("languages", []),
            "job_levels": entry.get("job_levels", []),
            "remote": entry.get("remote", ""),
            "adaptive": entry.get("adaptive", ""),
        }
        # Build a rich text blob for retrieval
        item["_text"] = _build_text(item)
        items.append(item)
    return items


def _build_text(item: Dict[str, Any]) -> str:
    parts = [
        item["name"],
        item["description"],
        " ".join(item["keys"]),
        " ".join(item["job_levels"]),
        " ".join(item["languages"]),
        item["duration"],
    ]
    return " ".join(p for p in parts if p).lower()


# ---------------------------------------------------------------------------
# TF-IDF retrieval
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class CatalogIndex:
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
        self._build_tfidf()

    def _build_tfidf(self):
        docs = [_tokenize(item["_text"]) for item in self.items]
        N = len(docs)

        # term -> document frequency
        df: Dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # idf
        self._idf: Dict[str, float] = {
            term: math.log((N + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

        # tf-idf vectors (sparse dicts)
        self._vectors: List[Dict[str, float]] = []
        for doc in docs:
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            vec = {
                term: (count / len(doc)) * self._idf.get(term, 1.0)
                for term, count in tf.items()
            }
            self._vectors.append(vec)

    def _query_vec(self, query: str) -> Dict[str, float]:
        tokens = _tokenize(query)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {
            term: (count / max(len(tokens), 1)) * self._idf.get(term, 1.0)
            for term, count in tf.items()
        }

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a.get(t, 0.0) * v for t, v in b.items())
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        qvec = self._query_vec(query)
        scores = [
            (self._cosine(qvec, dvec), i)
            for i, dvec in enumerate(self._vectors)
        ]
        scores.sort(reverse=True)
        return [self.items[i] for _, i in scores[:top_k]]

    def get_by_name(self, name: str) -> Dict[str, Any] | None:
        name_lower = name.lower()
        for item in self.items:
            if item["name"].lower() == name_lower:
                return item
        # fuzzy fallback
        for item in self.items:
            if name_lower in item["name"].lower():
                return item
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        return self.items


# Singleton
_catalog_items: List[Dict[str, Any]] | None = None
_catalog_index: CatalogIndex | None = None


def get_catalog() -> List[Dict[str, Any]]:
    global _catalog_items
    if _catalog_items is None:
        _catalog_items = load_catalog()
    return _catalog_items


def get_index() -> CatalogIndex:
    global _catalog_index
    if _catalog_index is None:
        _catalog_index = CatalogIndex(get_catalog())
    return _catalog_index
