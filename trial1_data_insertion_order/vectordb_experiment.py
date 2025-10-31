## Measuring accuracy and visited nodes per hop in HNSW (to examine the possibility of early termination)

# https://github.com/RyanLiGod/hnsw-python/blob/master/hnsw.py

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
import numpy as np
import struct
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict


class HNSW:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        try:
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except ValueError:
            print(a)
            print(b)

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_type, M=5, efConstruction=200, Mmax=None):
        if distance_type == "l2":
            distance_func = self.l2_distance
        elif distance_type == "cosine":
            distance_func = self.cosine_distance
        else:
            raise TypeError('Please check your distance type!')
        self.distance_func = distance_func
        self.vectorized_distance = self.vectorized_distance_
        self._M = M
        self._efConstruction = efConstruction
        self._Mmax = 2 * M if Mmax is None else Mmax
        self._level_mult = 1 / log2(M)
        self._graphs = []
        self._enter_point = None
        self.data = []
        self.visited_count = 0

        ##########
        self.visited_per_hop = []
        self.ann_per_hop = []
        #########

    ### Algorithm 1: INSERT
    def insert(self, q, efConstruction=None):

        if efConstruction is None:
            efConstruction = self._efConstruction

        distance = self.distance_func
        data = self.data
        graphs = self._graphs
        ep = self._enter_point
        M = self._M

        # line 4: determine level for the new element q
        l = int(-log2(random()) * self._level_mult) + 1
        idx = len(data)
        data.append(q)

        if ep is not None:
            neg_dist = -distance(q, data[ep])
            # distance(q, data[ep])|

            # line 5-7: find the closest neighbor for levels above the insertion level
            for lc in reversed(graphs[l:]):
                neg_dist, ep = self._search_layer(q, [(neg_dist, ep)], lc, 1)[0]

            # line 8-17: insert q at the relevant levels; W is a candidate list
            layer0 = graphs[0]
            W = [(neg_dist, ep)]  ## 추가

            for lc in reversed(graphs[:l]):
                M_layer = M if lc is not layer0 else self._Mmax

                # line 9: update W with the closest nodes found in the graph
                W = self._search_layer(q, W, lc, efConstruction)  ## 변경

                # line 10: insert the best neighbors for q at this layer
                lc[idx] = layer_idx = {}
                self._select(layer_idx, W, M_layer, lc, heap=True)

                # line 11-13: insert bidirectional links to the new node
                for j, dist in layer_idx.items():
                    self._select(lc[j], (idx, dist), M_layer, lc)

        # line 18: create empty graphs for all new levels
        for _ in range(len(graphs), l):
            graphs.append({idx: {}})
            self._enter_point = idx

    ### Algorithm 5: K-NN-SEARCH
    def search(self, q, K=5, efSearch=20):
        """Find the K points closest to q."""

        distance = self.distance_func
        graphs = self._graphs
        ep = self._enter_point
        self.visited_count = 0

        if ep is None:
            raise ValueError("Empty graph")

        neg_dist = -distance(q, self.data[ep])

        # line 1-5: search from top layers down to the second level
        for lc in reversed(graphs[1:]):
            neg_dist, ep = self._search_layer(q, [(neg_dist, ep)], lc, 1)[0]

        ##########
        self.visited_per_hop = []
        self.ann_per_hop = []
        ##########

        # line 6: search with efSearch neighbors at the bottom level
        W = self._search_layer(q, [(neg_dist, ep)], graphs[0], efSearch)

        if K is not None:
            W = nlargest(K, W)
        else:
            W.sort(reverse=True)

        return [(idx, -md) for md, idx in W]

    ### Algorithm 2: SEARCH-LAYER
    def _search_layer(self, q, W, lc, ef):

        vectorized_distance = self.vectorized_distance
        data = self.data

        # Step 1: Initialize candidate list and visited set
        C = [(-neg_dist, idx) for neg_dist, idx in W]
        heapify(C)
        heapify(W)
        visited = set(idx for _, idx in W)

        # Step 4-17: Explore neighbors until candidate list is exhausted
        while C:
            dist, c = heappop(C)
            furthest = -W[0][0]
            if dist > furthest:
                break
            neighbors = [e for e in lc[c] if e not in visited]
            visited.update(neighbors)
            dists = vectorized_distance(q, [data[e] for e in neighbors])
            for e, dist in zip(neighbors, dists):
                self.visited_count += 1
                neg_dist = -dist
                if len(W) < ef:
                    heappush(C, (dist, e))
                    heappush(W, (neg_dist, e))
                    furthest = -W[0][0]
                elif dist < furthest:
                    heappush(C, (dist, e))
                    heapreplace(W, (neg_dist, e))
                    furthest = -W[0][0]

            ##########
            self.visited_per_hop.append(len(visited))
            topk = nsmallest(min(ef, len(W)), ((-neg, idx) for neg, idx in W))  # (dist, id)
            self.ann_per_hop.append([idx for _, idx in topk])
            ##########

        return W

    ### Algorithm 3: SELECT-NEIGHBORS-SIMPLE
    def _select(self, R, C, M, lc, heap=False):

        if not heap:
            idx, dist = C
            if len(R) < M:
                R[idx] = dist
            else:
                max_idx, max_dist = max(R.items(), key=itemgetter(1))
                if dist < max_dist:
                    del R[max_idx]
                    R[idx] = dist
            return

        else:
            C = nlargest(M, C)
            R.update({idx: -neg_dist for neg_dist, idx in C})


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
test = read_fvecs(f"{base_path}/sift_query.fvecs")  # 10,000 × 128
neighbors = read_ivecs(f"{base_path}/sift_groundtruth.ivecs")  # 10,000 × 100

# Original dataset (1,000,000 × 128)
original_data = train  # Assuming `train` contains the original dataset
# Sampling size
# sample_size = 300_000
sample_size = 200_000
# Random sampling (set random seed for reproducibility)
np.random.seed(42)
sampled_indices = np.random.choice(len(original_data), size=sample_size, replace=False)
train = original_data[sampled_indices]

print("train:", train.shape, "test:", test.shape, "neighbors:", neighbors.shape)

def exact_topk_l2(train_subset, queries, K):
    out = np.empty((len(queries), K), dtype=np.int32)
    for i, q in enumerate(queries):
        d = np.sum((train_subset - q) ** 2, axis=1)  # L2^2
        idx = np.argpartition(d, K)[:K]  # top-K (unordered)
        idx = idx[np.argsort(d[idx])]  # sort by distance
        out[i] = idx
    return out

K_value = 10
# 예: 쿼리 1000개만 먼저
# random sample 1000 queries for faster processing
query_num = 1000
neighbors_subset = exact_topk_l2(train, test[:query_num], max(100, K_value))
# neighbors_subset = exact_topk_l2(train, test[np.random.choice(len(test), size=query_num, replace=False)], max(100, K_value))
# store neighbors_subset as a file for later use
with open('neighbors_subset.pkl', 'wb') as f:
    pickle.dump(neighbors_subset, f)

naivehnsw = HNSW("l2", M=16, efConstruction=200)
for i in range(len(train)):
    if i % 1000 == 0:
        print("Inserting data point:", i)
    naivehnsw.insert(train[i])

with open('naive_hnsw_model.pkl', 'wb') as f:
    pickle.dump(naivehnsw, f)

kmeans = KMeans(n_clusters=100, n_init='auto', random_state=21).fit(train)
labels = kmeans.labels_  # 각 벡터가 속한 클러스터 번호
centroids = kmeans.cluster_centers_

cluster_data = defaultdict(list)
for i, label in enumerate(labels):
    cluster_data[label].append((i, train[i]))  # Store a tuple of (original_index, data_point)

# # numpy 배열로 변환 - No longer needed as we need to preserve original indices
# cluster_data = {k: np.array(v) for k, v in cluster_data.items()}

hnswWithClusteredInput = HNSW("l2", M=16, efConstruction=200)
cluster_insertion_order = []
for i in range(len(cluster_data)):
    print("Inserting Cluster Number: ", i)
    # Iterate through the list of (original_index, data_point) tuples in each cluster
    for original_index, data_point in cluster_data[i]:
        cluster_insertion_order.append(int(original_index))  # Append the original index to the insertion order
        hnswWithClusteredInput.insert(data_point)

with open('cluster_insertion_order.pkl', 'wb') as f:
    pickle.dump(cluster_insertion_order, f)
with open('clustered_hnsw_model.pkl', 'wb') as f:
    pickle.dump(hnswWithClusteredInput, f)

hnswWithClusterRRInput = HNSW("l2", M=16, efConstruction=200)
cluster_rr_insertion_order = []
max_cluster_size = max(len(v) for v in cluster_data.values())
print("RR insertion start...")
for j in range(max_cluster_size):
    for i in range(len(cluster_data)):
        if j < len(cluster_data[i]):
            original_index, data_point = cluster_data[i][j]
            cluster_rr_insertion_order.append(int(original_index))  # Append the original index to the insertion order
            hnswWithClusterRRInput.insert(data_point)

with open('cluster_rr_insertion_order.pkl', 'wb') as f:
    pickle.dump(cluster_rr_insertion_order, f)
with open('clusteredRR_hnsw_model.pkl', 'wb') as f:
    pickle.dump(hnswWithClusterRRInput, f)

# Two-phase clustered insertion:
#  (Phase 1) seed insertion: insert a small fraction from each cluster in round-robin order
#  (Phase 2) bulk insertion: insert all remaining items cluster-by-cluster

# Hyperparameters for the two-phase strategy
seed_ratio = 0.2  # insert first 5% per cluster during Phase 1 (set to 0 to disable)
min_seed_per_cluster = 5  # ensure at least a few seeds per cluster
rng = np.random.default_rng(42)  # reproducible sampling inside each cluster

# Initialize index for the mixed strategy
hnswWithMixedClusteredInput = HNSW("l2", M=16, efConstruction=200)
mixed_cluster_insertion_order = []  # k-th inserted -> original train index (for id-space mapping)

# Build per-cluster arrays (copy from cluster_data to ensure deterministic order/shuffle per cluster)
clusters = sorted(cluster_data.keys())
per_cluster_items = {c: list(cluster_data[c]) for c in clusters}  # list of (orig_idx, vec)

# Shuffle within each cluster (optional but recommended to avoid within-cluster sorting bias)
for c in clusters:
    if len(per_cluster_items[c]) > 1:
        rng.shuffle(per_cluster_items[c])

# Determine seed sets and remaining sets per cluster
seed_sets = {}
remain_sets = {}
for c in clusters:
    n = len(per_cluster_items[c])
    seed_n = min(n, max(min_seed_per_cluster, int(n * seed_ratio)))
    seed_sets[c] = per_cluster_items[c][:seed_n]
    remain_sets[c] = per_cluster_items[c][seed_n:]

# -----------------
# Phase 1: round-robin insertion across clusters (seed points)
# -----------------
# Create iterators for each seed list
seed_iters = {c: iter(seed_sets[c]) for c in clusters}
inserted_seed_counts = defaultdict(int)
print("[MixedClustered] Phase 1: inserting seeds per cluster:", {c: len(seed_sets[c]) for c in clusters})
# Round-robin until all seed iters are exhausted
active = set(clusters)
while active:
    for c in list(active):
        try:
            orig_idx, vec = next(seed_iters[c])
            mixed_cluster_insertion_order.append(int(orig_idx))
            hnswWithMixedClusteredInput.insert(vec)
            inserted_seed_counts[c] += 1
        except StopIteration:
            active.remove(c)

# -----------------
# Phase 2: bulk insertion cluster-by-cluster (remaining points)
# -----------------
print("[MixedClustered] Phase 2: inserting remaining items per cluster:", {c: len(remain_sets[c]) for c in clusters})
for c in clusters:
    print("Inserting remaining items of Cluster Number: ", c)
    # Insert the remaining items of cluster c
    for orig_idx, vec in remain_sets[c]:
        mixed_cluster_insertion_order.append(int(orig_idx))
        hnswWithMixedClusteredInput.insert(vec)

print("[MixedClustered] Phase1(seeds) inserted per cluster:", {c: inserted_seed_counts[c] for c in clusters})
print("[MixedClustered] Total inserted:", len(mixed_cluster_insertion_order))

with open('mixed_cluster_insertion_order.pkl', 'wb') as f:
    pickle.dump(mixed_cluster_insertion_order, f)
with open('mixedClustered_hnsw_model.pkl', 'wb') as f:
    pickle.dump(hnswWithMixedClusteredInput, f)


def calculate_recall(true_neighbors, predicted_neighbors, K):
    K = int(K)
    """Calculates recall for a single query."""
    true_topk = set(map(int, true_neighbors[:K]))
    pred_topk = set(map(int, predicted_neighbors[:K]))
    return len(true_topk & pred_topk) / K

# data structure with key(string) and value for list of visited nodes per query
visited_nodes_per_query = {}
recall_per_query = {}

def last_or_zero(seq):
    return int(seq[-1]) if isinstance(seq, (list, tuple)) and len(seq) > 0 else 0

efSearch = [20, 40, 100, 200, 400, 800]

for ef in efSearch:

    naive_visited_nodes = []
    naive_recall = []

    print("Evaluating efSearch =", ef)
    print("Naive Insertion Search with efSearch =", ef)
    K_value = 10  # Assuming K=10 for recall calculation as per search results
    for i in range(query_num):
        query = test[i]
        query_true_neighbors = neighbors_subset[i]
        naive_search_results = naivehnsw.search(query, K=K_value, efSearch=ef)
        naive_visited_nodes.append(list(naivehnsw.visited_per_hop))  # Store a copy of visited_per_hop

        naive_search_results_indices = [idx for idx, dist in naive_search_results][:K_value]
        naive_recall.append(calculate_recall(query_true_neighbors, naive_search_results_indices, K_value))

    clustered_visited_nodes = []
    clustered_recall = []
    print("Clustered Insertion Search with efSearch =", ef)

    K_value = 10  # Assuming K=10 for recall calculation as per search results
    for i in range(query_num):
        query = test[i]
        query_true_neighbors = neighbors_subset[i]
        clustered_search_results = hnswWithClusteredInput.search(query, K=K_value, efSearch=ef)
        clustered_visited_nodes.append(list(hnswWithClusteredInput.visited_per_hop))  # Store a copy of visited_per_hop

        clustered_search_indices = [idx for idx, dist in clustered_search_results]
        clustered_search_orig_ids = [cluster_insertion_order[idx] for idx in clustered_search_indices][:K_value]
        clustered_recall.append(calculate_recall(query_true_neighbors, clustered_search_orig_ids, K_value))

    print("Clustered Round-Robin Insertion Search with efSearch =", ef)
    clustered_RR_visited_nodes = []
    clustered_RR_recall = []

    K_value = 10  # Assuming K=10 for recall calculation as per search results
    for i in range(query_num):
        query = test[i]
        query_true_neighbors = neighbors_subset[i]
        clustered_RR_search_results = hnswWithClusterRRInput.search(query, K=K_value, efSearch=ef)
        clustered_RR_visited_nodes.append(
            list(hnswWithClusterRRInput.visited_per_hop))  # Store a copy of visited_per_hop

        clustered_RR_search_indices = [idx for idx, dist in clustered_RR_search_results]
        clustered_RR_search_orig_ids = [cluster_rr_insertion_order[idx] for idx in clustered_RR_search_indices][
            :K_value]
        clustered_RR_recall.append(calculate_recall(query_true_neighbors, clustered_RR_search_orig_ids, K_value))

    print("Mixed Clustered Insertion Search with efSearch =", ef)
    mixed_clustered_visited_nodes = []
    mixed_clustered_recall = []

    K_value = 10  # Assuming K=10 for recall calculation as per search results
    for i in range(query_num):
        query = test[i]
        query_true_neighbors = neighbors_subset[i]
        mixed_clustered_search_results = hnswWithMixedClusteredInput.search(query, K=K_value, efSearch=ef)
        mixed_clustered_visited_nodes.append(
            list(hnswWithMixedClusteredInput.visited_per_hop))  # Store a copy of visited_per_hop

        mixed_clustered_search_indices = [idx for idx, dist in mixed_clustered_search_results]
        mixed_clustered_search_orig_ids = [mixed_cluster_insertion_order[idx] for idx in
                                           mixed_clustered_search_indices][:K_value]
        mixed_clustered_recall.append(calculate_recall(query_true_neighbors, mixed_clustered_search_orig_ids, K_value))

    # total unique visited nodes per query
    naive_total = [last_or_zero(v) for v in naive_visited_nodes]
    clustered_total = [last_or_zero(v) for v in clustered_visited_nodes]
    clustered_rr_total = [last_or_zero(v) for v in clustered_RR_visited_nodes]
    mixed_total = [last_or_zero(v) for v in
                   mixed_clustered_visited_nodes] if 'mixed_clustered_visited_nodes' in globals() else None

    # store visited_node results for this ef
    visited_nodes_per_query[f"naive_ef{ef}"] = np.asarray(naive_total, dtype=np.int64).mean()
    visited_nodes_per_query[f"clustered_ef{ef}"] = np.asarray(clustered_total, dtype=np.int64).mean()
    visited_nodes_per_query[f"clusteredRR_ef{ef}"] = np.asarray(clustered_rr_total, dtype=np.int64).mean()
    if mixed_total is not None:
        visited_nodes_per_query[f"mixed_ef{ef}"] = np.asarray(mixed_total, dtype=np.int64).mean()
    print("Visited Nodes (mean) so far:", visited_nodes_per_query)

    # store recall results for this ef
    recall_per_query[f"naive_ef{ef}"] = np.asarray(naive_recall, dtype=np.float32).mean()
    recall_per_query[f"clustered_ef{ef}"] = np.asarray(clustered_recall, dtype=np.float32).mean()
    recall_per_query[f"clusteredRR_ef{ef}"] = np.asarray(clustered_RR_recall, dtype=np.float32).mean()
    if mixed_clustered_recall is not None:
        recall_per_query[f"mixed_ef{ef}"] = np.asarray(mixed_clustered_recall, dtype=np.float32).mean()
    print("Recall (mean) so far:", recall_per_query)

# Show results as a table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
results_df = pd.DataFrame({
    "Visited Nodes": visited_nodes_per_query,
    "Recall": recall_per_query
}).T
results_df.index.name = "Method"
results_df = results_df.reset_index()
results_df = results_df.sort_values(by=["Method"])
print(results_df)
