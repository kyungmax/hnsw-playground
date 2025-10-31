## Measuring accuracy and visited nodes per hop in HNSW (to examine the possibility of early termination)

# https://github.com/RyanLiGod/hnsw-python/blob/master/hnsw.py

import struct
from collections import defaultdict

import hnswlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

## Load SIFT1M dataset (.fvecs / .ivecs)

def read_fvecs(filename):
    """Reads .fvecs binary file into np.ndarray of shape (n, d)."""
    with open(filename, 'rb') as f:
        data = f.read()
    dim = struct.unpack('i', data[:4])[0]
    vecs = np.frombuffer(data, dtype=np.float32)
    vecs = vecs.reshape(-1, dim + 1)[:, 1:]  # drop the leading 'dim'
    return vecs


def read_ivecs(filename):
    """Reads .ivecs binary file into np.ndarray of shape (n, k)."""
    with open(filename, 'rb') as f:
        data = f.read()
    dim = struct.unpack('i', data[:4])[0]
    vecs = np.frombuffer(data, dtype=np.int32)
    vecs = vecs.reshape(-1, dim + 1)[:, 1:]
    return vecs


# 데이터셋 경로 (현재 구조에 맞춰 수정)
base_path = "../datasets"
train = read_fvecs(f"{base_path}/sift_base.fvecs")  # 1,000,000 × 128
print(f"Loaded train dataset with shape: {train.shape}")
test = read_fvecs(f"{base_path}/sift_query.fvecs")  # 10,000 × 128
print(f"Loaded test dataset with shape: {test.shape}")
neighbors = read_ivecs(f"{base_path}/sift_groundtruth.ivecs")  # 10,000 × 100
print(f"Loaded neighbors dataset with shape: {neighbors.shape}")

dim = train.shape[1]
efConstruction = 200
paramM = 16

def build_hnsw_index(order, M=16, efc=200, space='l2'):
    """Build an hnswlib index by adding items in a specific order. ids are original indices."""
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=len(train), ef_construction=efc, M=M)
    # add in batches to keep order while being efficient
    B = 20000
    total_added = 0
    for s in range(0, len(order), B):
        chunk_idx = order[s:s+B]
        print(f"Adding batch: {s} to {s+len(chunk_idx)} (total so far: {total_added})")
        p.add_items(train[chunk_idx], ids=np.array(chunk_idx, dtype=np.int32))
        total_added += len(chunk_idx)
        print(f"Finished adding batch: {s} to {s+len(chunk_idx)} (total added: {total_added})")
    return p

K_value = 10

# ---- Build indices with hnswlib (order matters via add_items batching) ----
# 1) naive(random) insertion order
# print("Building naive/random order index ...")
# rng_global = np.random.default_rng(42)
# naive_order = rng_global.permutation(len(train)).tolist()
# naive_index = build_hnsw_index(naive_order, M=paramM, efc=efConstruction, space='l2')

# 2) k-means clustering to derive clustered orders
kmeans = KMeans(n_clusters=1000, n_init='auto', random_state=21).fit(train)
labels = kmeans.labels_
cluster_data = defaultdict(list)
for i, lbl in enumerate(labels):
    cluster_data[int(lbl)].append(i)
#
# # 2-1) strict clustered order (cluster by cluster)
# print("Building clustered order index ...")
# clustered_order = []
# for c in sorted(cluster_data.keys()):
#     clustered_order.extend(cluster_data[c])
# clustered_index = build_hnsw_index(clustered_order, M=paramM, efc=efConstruction, space='l2')
#
# # 2-2) round-robin across clusters
# print("Building clustered round-robin order index ...")
# max_len = max(len(v) for v in cluster_data.values())
# rr_order = []
# for j in range(max_len):
#     for c in sorted(cluster_data.keys()):
#         if j < len(cluster_data[c]):
#             rr_order.append(cluster_data[c][j])
# clustered_rr_index = build_hnsw_index(rr_order, M=paramM, efc=efConstruction, space='l2')
#
# # 2-3) two-phase (seed round-robin then bulk per cluster)
# print("Building mixed two-phase order index ...")
# seed_ratio = 0.2
# min_seed = 5
# per_cluster = {c: cluster_data[c][:] for c in cluster_data}
# # shuffle within cluster to avoid within-cluster bias
# rng_shuffle = np.random.default_rng(42)
# for c in per_cluster:
#     if len(per_cluster[c]) > 1:
#         rng_shuffle.shuffle(per_cluster[c])
# # split to seeds/remains
# seeds = {}
# remains = {}
# for c, items in per_cluster.items():
#     n = len(items)
#     sn = min(n, max(min_seed, int(n * seed_ratio)))
#     seeds[c] = items[:sn]
#     remains[c] = items[sn:]
# # round-robin seeds
# seed_iters = {c: iter(seeds[c]) for c in sorted(seeds.keys())}
# active = set(seed_iters.keys())
# mixed_order = []
# while active:
#     for c in list(active):
#         try:
#             mixed_order.append(next(seed_iters[c]))
#         except StopIteration:
#             active.remove(c)
# # bulk remains per cluster
# for c in sorted(remains.keys()):
#     mixed_order.extend(remains[c])
# mixed_index = build_hnsw_index(mixed_order, M=paramM, efc=efConstruction, space='l2')

# 2-4) reverse two-phase: bulk per cluster first (seed portion), then round-robin for the remaining
print("Building bulk-then-roundrobin order index ...")
rev_seed_ratio = 0.7
rev_min_seed = 5
per_cluster_rev = {c: cluster_data[c][:] for c in cluster_data}
# shuffle within cluster to avoid within-cluster bias
rng_shuffle_rev = np.random.default_rng(43)
for c in per_cluster_rev:
    if len(per_cluster_rev[c]) > 1:
        rng_shuffle_rev.shuffle(per_cluster_rev[c])
# split to seeds(remain the same naming): here 'seeds' are the initial bulk inserted per cluster
rev_seeds = {}
rev_remains = {}
for c, items in per_cluster_rev.items():
    n = len(items)
    sn = min(n, max(rev_min_seed, int(n * rev_seed_ratio)))
    rev_seeds[c] = items[:sn]
    rev_remains[c] = items[sn:]
# Phase A: bulk insert the seed portion per cluster
bulk_then_rr_order = []
for c in sorted(rev_seeds.keys()):
    bulk_then_rr_order.extend(rev_seeds[c])
# Phase B: round-robin the remaining items
if len(rev_remains) > 0:
    max_len_rev = max((len(v) for v in rev_remains.values()), default=0)
    for j in range(max_len_rev):
        for c in sorted(rev_remains.keys()):
            if j < len(rev_remains[c]):
                bulk_then_rr_order.append(rev_remains[c][j])

bulk_then_rr_index = build_hnsw_index(bulk_then_rr_order, M=paramM, efc=efConstruction, space='l2')


def calculate_recall(true_neighbors, predicted_neighbors, K):
    K = int(K)
    """Calculates recall for a single query."""
    true_topk = set(map(int, true_neighbors[:K]))
    pred_topk = set(map(int, predicted_neighbors[:K]))
    return len(true_topk & pred_topk) / K

import time

def eval_hnswlib(index, test_queries, gt_rows, ef_list, K=10):
    qps = {}
    rec = {}
    nQ = len(test_queries)
    for ef in ef_list:
        print(f"Starting evaluation for ef={ef} ...")
        index.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = index.knn_query(test_queries, k=K)
        t1 = time.perf_counter()
        # mean recall over queries
        r = []
        for i in range(nQ):
            r.append(calculate_recall(gt_rows[i], labels[i], K))
        avg_recall = float(np.mean(r))
        qps[f"ef{ef}"] = float(nQ / (t1 - t0 + 1e-12))
        rec[f"ef{ef}"] = avg_recall
        print(f"Finished evaluation for ef={ef}: Average Recall={avg_recall:.4f}, QPS={qps[f'ef{ef}']:.2f}")
    return qps, rec

efSearch = [20, 40, 100, 200, 400, 800]

# print("Evaluating naive/random order ...")
# naive_qps, naive_rec = eval_hnswlib(naive_index, test, neighbors, efSearch, K=K_value)
#
# print("Evaluating clustered order ...")
# clustered_qps, clustered_rec = eval_hnswlib(clustered_index, test, neighbors, efSearch, K=K_value)
#
# print("Evaluating clustered round-robin order ...")
# rr_qps, rr_rec = eval_hnswlib(clustered_rr_index, test, neighbors, efSearch, K=K_value)
#
# print("Evaluating mixed two-phase order ...")
# mixed_qps, mixed_rec = eval_hnswlib(mixed_index, test, neighbors, efSearch, K=K_value)

print("Evaluating bulk-then RR order ...")
bulk_then_rr_qps, bulk_then_rr_rec = eval_hnswlib(bulk_then_rr_index, test, neighbors, efSearch, K=K_value)

# Assemble results table (QPS + Recall)
print("Final Results Table")
rows = []
for tag, q, r in [
    # ("naive", naive_qps, naive_rec),
    # ("clustered", clustered_qps, clustered_rec),
    # ("clusteredRR", rr_qps, rr_rec),
    # ("mixed", mixed_qps, mixed_rec),
    ("bulkThenRR", bulk_then_rr_qps, bulk_then_rr_rec),
]:
    for ef in efSearch:
        rows.append({
            "Method": f"{tag}_ef{ef}",
            "QPS": q[f"ef{ef}"],
            "Recall": r[f"ef{ef}"]
        })
results_df = pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)
print(results_df)
