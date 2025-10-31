## Measuring accuracy and visited nodes per hop in HNSW (to examine the possibility of early termination)

# https://github.com/RyanLiGod/hnsw-python/blob/master/hnsw.py

import struct
from collections import defaultdict

import hnswlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import h5py
import os

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
base_path = "./datasets"

# train = read_fvecs(f"{base_path}/sift_base.fvecs")  # 1,000,000 × 128
# print(f"Loaded train dataset with shape: {train.shape}")
# test = read_fvecs(f"{base_path}/sift_query.fvecs")  # 10,000 × 128
# print(f"Loaded test dataset with shape: {test.shape}")
# neighbors = read_ivecs(f"{base_path}/sift_groundtruth.ivecs")  # 10,000 × 100
# print(f"Loaded neighbors dataset with shape: {neighbors.shape}")

# now, I want to use ‘glove-200-angular.hdf5’

# 데이터셋 경로
file_path = base_path + "/glove-200-angular.hdf5"

# h5py를 사용하여 파일 열기
with h5py.File(file_path, 'r') as f:
    # HDF5 파일 내의 데이터셋 키 확인 (어떤 데이터가 있는지 모를 경우 유용)
    print(f"Keys in HDF5 file: {list(f.keys())}")

    # 각 데이터셋을 numpy 배열로 불러오기
    train = np.array(f['train'])
    test = np.array(f['test'])
    neighbors = np.array(f['neighbors'])
    # distances 데이터셋이 있다면 같이 로드할 수 있습니다.
    # distances = np.array(f['distances'])

# random sample 100,000 from train
seed = 42
n_target = 10_000
rng = np.random.RandomState(seed)
idx = rng.choice(train.shape[0], n_target, replace=False)
train = train[idx]


dim = train.shape[1]
efConstruction = 400
paramM = 4
distance_method = 'cosine'

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
print("Building naive/random order index ...")
rng_global = np.random.default_rng(42)
naive_order = rng_global.permutation(len(train)).tolist()
naive_index = build_hnsw_index(naive_order, M=paramM, efc=efConstruction, space=distance_method)
print("Naive Index Node Count: ", naive_index.get_current_count())
naive_index_path = "index/naive_hnsw_index.index"
naive_index.save_index(naive_index_path)
print("Naive Index File Size (bytes): ",os.path.getsize(naive_index_path))

# 2) k-means clustering to derive clustered orders
kmeans = KMeans(n_clusters=500, n_init='auto', random_state=21).fit(train)
labels = kmeans.labels_
cluster_data = defaultdict(list)
for i, lbl in enumerate(labels):
    cluster_data[int(lbl)].append(i)

# 2-1) strict clustered order (cluster by cluster)
print("Building clustered order index ...")
clustered_order = []
for c in sorted(cluster_data.keys()):
    clustered_order.extend(cluster_data[c])
clustered_index = build_hnsw_index(clustered_order, M=paramM, efc=efConstruction, space=distance_method)
print("Clustered Index Node Count: ", clustered_index.get_current_count())
clusted_index_path = "index/clustered_hnsw_index.index"
clustered_index.save_index(clusted_index_path)
print("Clustered Index File Size (bytes): ", os.path.getsize(clusted_index_path))

# 2-2) round-robin across clusters
print("Building clustered round-robin order index ...")
max_len = max(len(v) for v in cluster_data.values())
rr_order = []
for j in range(max_len):
    for c in sorted(cluster_data.keys()):
        if j < len(cluster_data[c]):
            rr_order.append(cluster_data[c][j])
clustered_rr_index = build_hnsw_index(rr_order, M=paramM, efc=efConstruction, space=distance_method)
print("Clustered RR Index Node Count: ", clustered_rr_index.get_current_count())
clustered_rr_index_path = "index/clustered_rr_index.index"
clustered_rr_index.save_index(clustered_rr_index_path)
print("Clustered RR Index File Size (bytes): ", os.path.getsize(clustered_rr_index_path))

# 2-3) two-phase (seed round-robin then bulk per cluster)
print("Building mixed two-phase order index for multiple seed_ratio ...")
seed_ratios = [0.2]
min_seed = 5
mixed_index = []

for seed_ratio in seed_ratios:
    per_cluster = {c: cluster_data[c][:] for c in cluster_data}
    rng_shuffle = np.random.default_rng(42)
    for c in per_cluster:
        if len(per_cluster[c]) > 1:
            rng_shuffle.shuffle(per_cluster[c])
    seeds = {}
    remains = {}
    for c, items in per_cluster.items():
        n = len(items)
        sn = min(n, max(min_seed, int(n * seed_ratio)))
        seeds[c] = items[:sn]
        remains[c] = items[sn:]
    seed_iters = {c: iter(seeds[c]) for c in sorted(seeds.keys())}
    active = set(seed_iters.keys())
    mixed_order = []
    while active:
        for c in list(active):
            try:
                mixed_order.append(next(seed_iters[c]))
            except StopIteration:
                active.remove(c)
    for c in sorted(remains.keys()):
        mixed_order.extend(remains[c])
    mixed_index.append(build_hnsw_index(
        mixed_order, M=paramM, efc=efConstruction, space=distance_method
    ))
    print("Mixed Index Node Count: ", mixed_index[0].get_current_count())
    mixed_index_path = f"./mixed_hnsw_index_seedratio_{seed_ratio}.index"
    mixed_index[-1].save_index(mixed_index_path)
    print("Mixed Index File Size (bytes): ", os.path.getsize(mixed_index_path))
    # 결과 저장 또는 분석
    print(f"seed_ratio={seed_ratio}에 대해 mixed_index 생성 완료")

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

print("Evaluating naive/random order ...")
naive_qps, naive_rec = eval_hnswlib(naive_index, test, neighbors, efSearch, K=K_value)

print("Evaluating clustered order ...")
clustered_qps, clustered_rec = eval_hnswlib(clustered_index, test, neighbors, efSearch, K=K_value)

print("Evaluating clustered round-robin order ...")
rr_qps, rr_rec = eval_hnswlib(clustered_rr_index, test, neighbors, efSearch, K=K_value)

print("Evaluating mixed two-phase order ...")
for idx, seed_ratio in enumerate(seed_ratios):
    print(f"Evaluating mixed two-phase order with seed_ratio={seed_ratio} ...")
    mixed_qps, mixed_rec = eval_hnswlib(mixed_index[idx], test, neighbors, efSearch, K=K_value)
    print(f"Results for seed_ratio={seed_ratio}:")
    for ef in efSearch:
        print(f"  ef={ef}: QPS={mixed_qps[f'ef{ef}']:.2f}, Recall={mixed_rec[f'ef{ef}']:.4f}")
# mixed_qps, mixed_rec = eval_hnswlib(mixed_index, test, neighbors, efSearch, K=K_value)
# Assemble results table (QPS + Recall)
print("Final Results Table")
rows = []
for tag, q, r in [
    ("naive", naive_qps, naive_rec),
    ("clustered", clustered_qps, clustered_rec),
    ("clusteredRR", rr_qps, rr_rec),
    ("mixed", mixed_qps, mixed_rec),
]:
    for ef in efSearch:
        rows.append({
            "Method": f"{tag}_ef{ef}",
            "QPS": q[f"ef{ef}"],
            "Recall": r[f"ef{ef}"]
        })
results_df = pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)
print(results_df)
