from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scisum_qwen.evidence.chunker import PaperChunk


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk: PaperChunk
    score: float
    rank: int


class EvidenceIndex:
    def __init__(self, chunks: list[PaperChunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)

    @classmethod
    def from_chunks(cls, chunks: list[PaperChunk], embedder) -> "EvidenceIndex":
        texts = [chunk.text for chunk in chunks]
        embedder.fit(texts)
        embeddings = embedder.encode(texts)
        return cls(chunks, embeddings)

    def search(self, query_text: str, embedder, *, top_k: int = 3) -> list[RetrievedEvidence]:
        query_vector = embedder.encode([query_text])[0]
        scores = self.embeddings @ query_vector
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedEvidence(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank + 1,
            )
            for rank, index in enumerate(ranked_indices)
        ]

