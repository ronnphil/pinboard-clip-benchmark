# Pinboard retrieval benchmark

Measures how well CLIP ViT-B/32 retrieves images from text, using COCO val2017 captions.
This is the retrieval core for a visual-discovery app (pins, boards, "more like this").
Only the benchmark exists so far. The app itself is not built yet.

## Result (1,000 images, seed 0)

| Metric | Value |
|---|---|
| Images indexed | 1,000 (random COCO val2017 subset) |
| Queries | 5,002 (every caption is a query, its source image is the right answer) |
| Recall@1 | 49.6% |
| **Recall@5** | **78.4%** |
| Recall@10 | 88.0% |
| Search latency | p50 0.34 ms, p95 1.02 ms |

- Latency is the vector search only (brute-force cosine over precomputed embeddings,
  top 5, 500 queries, CPU). It excludes encoding the query text and the images.
- A 1,000-image subset is easier than the full 5,000-image COCO test split, so these
  recall numbers are not comparable to full-split figures in papers.
- Exact values are in `results.json`.

## Reproduce

```bash
pip install sentence-transformers pillow numpy pytest
# captions (about 250 MB zip, only one file is needed)
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/captions_val2017.json
python eval_coco.py 1000     # downloads the 1,000 sampled images, embeds, evaluates
```

The first run embeds 1,000 images on CPU, which takes a couple of minutes.

## Tests

```bash
python -m pytest -q          # 24 tests, no model download needed
```

Tests cover the pure retrieval logic in `retrieval.py`: seeded sampling, query building,
normalization, rank and recall@k (including ties), and top-k search.

## Layout

- `retrieval.py` NumPy-only helpers (sampling, queries, rank, recall@k, top-k)
- `eval_coco.py` runs the benchmark end to end
- `tests/` unit tests
- `results.json` last measured results
