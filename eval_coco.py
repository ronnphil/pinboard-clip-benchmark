"""CLIP ViT-B/32 text->image retrieval on a COCO val2017 subset.

Usage: python eval_coco.py [N_IMAGES]   (default 1000)

Each image has ~5 captions. Every caption is a query and the image it came from
is the correct answer. Reports recall@1/5/10 and search latency (embeddings
precomputed, so latency covers the vector search only, not text/image encoding).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from retrieval import build_queries, rank_of_truth, recall_at_k, sample_ids, top_k

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 0
CAPTIONS = "annotations/captions_val2017.json"

if not os.path.exists(CAPTIONS):
    sys.exit("Missing captions. See README: download annotations_trainval2017.zip and extract "
             "annotations/captions_val2017.json")

data = json.load(open(CAPTIONS))
caps = {}
for a in data["annotations"]:
    caps.setdefault(a["image_id"], []).append(a["caption"].strip())
ids = sample_ids(caps, N, SEED)

os.makedirs("imgs", exist_ok=True)


def fetch(i):
    p = f"imgs/{i:012d}.jpg"
    if not os.path.exists(p):
        with urlopen(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", timeout=30) as r:
            open(p, "wb").write(r.read())
    return p


with ThreadPoolExecutor(16) as ex:
    paths = list(ex.map(fetch, ids))

model = SentenceTransformer("clip-ViT-B-32")
imgs = [Image.open(p).convert("RGB") for p in paths]
img_emb = model.encode(imgs, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)

queries, truth = build_queries(ids, caps)
q_emb = model.encode(queries, batch_size=128, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)

ranks = rank_of_truth(q_emb @ img_emb.T, truth)
res = {f"recall@{k}": recall_at_k(ranks, k) for k in (1, 5, 10)}

lat = []
for q in q_emb[:500]:
    t0 = time.perf_counter()
    top_k(img_emb, q, 5)
    lat.append((time.perf_counter() - t0) * 1000)
res.update(images=len(ids), queries=len(queries),
           search_ms_p50=float(np.percentile(lat, 50)),
           search_ms_p95=float(np.percentile(lat, 95)),
           seed=SEED, model="clip-ViT-B-32")
print(json.dumps(res, indent=2))
json.dump(res, open("results.json", "w"), indent=2)
