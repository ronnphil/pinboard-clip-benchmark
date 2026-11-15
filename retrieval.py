"""Pure retrieval helpers (NumPy only, no model needed), so they can be unit-tested."""
import random

import numpy as np


def sample_ids(caps: dict, n: int, seed: int = 0) -> list:
    """Reproducible random subset of image ids that have captions."""
    ids = sorted(caps)
    random.Random(seed).shuffle(ids)
    return ids[:n]


def build_queries(ids: list, caps: dict):
    """Every caption of every image is a query. Returns (queries, truth_index)."""
    queries, truth = [], []
    for idx, i in enumerate(ids):
        for c in caps[i]:
            queries.append(c)
            truth.append(idx)
    return queries, np.array(truth)


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def rank_of_truth(sims: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """0-based rank of the correct image for each query (0 = top result).

    Ties count in the correct image's favour only if strictly lower scores,
    i.e. rank = number of images scoring strictly higher.
    """
    correct = sims[np.arange(len(truth)), truth]
    return (sims > correct[:, None]).sum(axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Percentage of queries whose correct image is in the top k."""
    return float((ranks < k).mean() * 100)


def top_k(index: np.ndarray, query: np.ndarray, k: int = 5) -> np.ndarray:
    """Indices of the k most similar rows of `index`, best first (cosine on unit vectors)."""
    k = min(k, len(index))
    scores = index @ query
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]
