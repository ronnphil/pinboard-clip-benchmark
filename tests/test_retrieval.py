import numpy as np
import pytest

from retrieval import build_queries, normalize, rank_of_truth, recall_at_k, sample_ids, top_k

CAPS = {10: ["a cat", "a kitten"], 20: ["a dog"], 30: ["a car", "red car", "a truck"], 40: ["a boat"]}


# sample_ids
def test_sample_ids_is_reproducible():
    assert sample_ids(CAPS, 3, seed=0) == sample_ids(CAPS, 3, seed=0)


def test_sample_ids_respects_n():
    assert len(sample_ids(CAPS, 2)) == 2


def test_sample_ids_n_larger_than_pool():
    assert sorted(sample_ids(CAPS, 99)) == sorted(CAPS)


def test_sample_ids_different_seeds_can_differ():
    big = {i: ["x"] for i in range(100)}
    assert sample_ids(big, 10, seed=0) != sample_ids(big, 10, seed=1)


def test_sample_ids_has_no_duplicates():
    out = sample_ids(CAPS, 4)
    assert len(set(out)) == len(out)


# build_queries
def test_build_queries_one_query_per_caption():
    queries, truth = build_queries([10, 20, 30], CAPS)
    assert len(queries) == 6 and len(truth) == 6


def test_build_queries_truth_points_at_source_image():
    queries, truth = build_queries([10, 20, 30], CAPS)
    assert truth.tolist() == [0, 0, 1, 2, 2, 2]
    assert queries[2] == "a dog"


def test_build_queries_empty():
    queries, truth = build_queries([], CAPS)
    assert queries == [] and len(truth) == 0


# normalize
def test_normalize_gives_unit_vectors():
    out = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_zero_vector_does_not_nan():
    assert not np.isnan(normalize(np.zeros((1, 3)))).any()


# rank_of_truth
def test_rank_of_truth_top_hit_is_zero():
    sims = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    assert rank_of_truth(sims, np.array([0, 1])).tolist() == [0, 0]


def test_rank_of_truth_counts_better_images():
    sims = np.array([[0.1, 0.9, 0.5]])  # correct (idx 0) is worst
    assert rank_of_truth(sims, np.array([0])).tolist() == [2]


def test_rank_of_truth_ties_do_not_count_against():
    sims = np.array([[0.5, 0.5, 0.5]])
    assert rank_of_truth(sims, np.array([1])).tolist() == [0]


# recall_at_k
def test_recall_at_k_basic():
    ranks = np.array([0, 1, 4, 5, 20])
    assert recall_at_k(ranks, 1) == 20.0
    assert recall_at_k(ranks, 5) == 60.0
    assert recall_at_k(ranks, 10) == 80.0


def test_recall_at_k_perfect_and_zero():
    assert recall_at_k(np.zeros(4, dtype=int), 1) == 100.0
    assert recall_at_k(np.full(4, 9), 5) == 0.0


def test_recall_is_monotonic_in_k():
    ranks = np.random.default_rng(0).integers(0, 50, 200)
    vals = [recall_at_k(ranks, k) for k in (1, 5, 10, 50)]
    assert vals == sorted(vals)


# top_k
def test_top_k_orders_best_first():
    index = normalize(np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]))
    q = normalize(np.array([[1.0, 0.1]]))[0]
    assert top_k(index, q, 3).tolist() == [0, 1, 2]


def test_top_k_returns_k_items():
    index = normalize(np.random.default_rng(1).normal(size=(50, 8)))
    assert len(top_k(index, index[0], 5)) == 5


def test_top_k_larger_than_index_is_clamped():
    index = normalize(np.random.default_rng(2).normal(size=(3, 4)))
    assert len(top_k(index, index[0], 10)) == 3


def test_top_k_self_query_finds_itself_first():
    index = normalize(np.random.default_rng(3).normal(size=(30, 16)))
    for i in (0, 7, 29):
        assert top_k(index, index[i], 5)[0] == i


# end-to-end on synthetic embeddings
def test_perfect_embeddings_give_full_recall():
    img = normalize(np.eye(5))
    truth = np.arange(5)
    assert recall_at_k(rank_of_truth(img @ img.T, truth), 1) == 100.0


@pytest.mark.parametrize("n", [1, 2, 10])
def test_pipeline_shapes(n):
    ids = sample_ids({i: ["a", "b"] for i in range(20)}, n)
    queries, truth = build_queries(ids, {i: ["a", "b"] for i in range(20)})
    assert len(queries) == 2 * n and truth.max() == n - 1
