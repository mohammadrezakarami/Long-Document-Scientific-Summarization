from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BagOfWordsEmbedder:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self._is_fit = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._is_fit = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("BagOfWordsEmbedder must be fit before encoding.")
        matrix = self.vectorizer.transform(texts).astype(np.float32).toarray()
        return _normalize_rows(matrix)


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def fit(self, texts: list[str]) -> None:
        del texts

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return matrix.astype(np.float32)


def build_embedder(preferred_model: str | None = None, *, allow_fallback: bool = True):
    if preferred_model:
        try:
            return SentenceTransformerEmbedder(preferred_model)
        except Exception:
            if not allow_fallback:
                raise
    return BagOfWordsEmbedder()

