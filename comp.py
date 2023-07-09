import numpy as np
import time
from sklearn.cluster import KMeans
import pickle

from typing import Union, List
import argparse

def_n_dim = 2000 
def_n_clusters = 4
def_nominal_duration = 4
def_duration_var = 1.5
def_total_duration = 40
def_duration_tolerance_ratio = 0.15
def_alphas = [0, 0.001, 0.01, 0.1, 1, 10]
def_percentile_list = [99, 95, 90, 75, 50]
def_seed = 0

def get_cos_sum(
        hit_pattern: List[int],
        cos_matrix: np.ndarray,
) -> float:

    n_vecs = cos_matrix.shape[0]
    hit_idx = [i for i in range(n_vecs) if hit_pattern[i]>0]
    sum_cos = 0
    for i, idx in enumerate(hit_idx):
        if i+1<len(hit_idx):
            for j in range(i+1,len(hit_idx)):
                jdx = hit_idx[j]
                sum_cos += cos_matrix[idx,jdx]

    return sum_cos
    

def main(
        n_vecs: int, 
        n_dim: int = def_n_dim,
        n_clusters: int = def_n_clusters,
        nominal_duration: float = def_nominal_duration,
        duration_var: float = def_duration_var,
        total_duration: float = def_total_duration,
        duration_tolerance_ratio: float = def_duration_tolerance_ratio,
        alphas: Union[float,List[float]] = def_alphas,
        percentile_list: List[Union[int,float]] = def_percentile_list,
        seed: int = def_seed,
):

    rng = np.random.default_rng(seed)

    if not isinstance(alphas,list):
        alphas = [alphas]
        
    vecs = rng.normal(size=(n_vecs,n_dim))
    scores = rng.uniform(size=(n_vecs,))
    durations = rng.normal(loc=nominal_duration,scale=duration_var,size=(n_vecs,))
    print(f"vecs.shape={vecs.shape}")
    print(f"scores={scores}")
    print(f"durations={durations}")
    print(f"nominal_duration={nominal_duration}")

    print("----------")
    start_cos = time.time()
    norms = np.sqrt(np.sum(vecs*vecs,axis=1))
    print(f"norms.shape={norms.shape}")
    norm_time = time.time() - start_cos
    all_pair_cos = np.matmul(
        (vecs.T/norms).T,
        (vecs.T/norms),
    )
    matmul_time = time.time() - start_cos - norm_time
    print(f"all_pair_cos.shape={all_pair_cos.shape}")
    print(all_pair_cos)
    print(f"norm_time={norm_time}, matmul_time={matmul_time}")

    print()
    print("----------")
    start_kmeans = time.time()
    kmeans = KMeans(n_clusters=n_clusters)
    pred = kmeans.fit_predict(vecs)
    kmeans_time = time.time() - start_kmeans
    print(pred)
    print(f"kmeans_time={kmeans_time}")

    print("---------- kmeans selection")
    start_kmeans_selection = time.time()
    indices = {}
    for i_cluster in range(n_clusters):
        cluster_indices = [i for i in range(n_vecs) if pred[i]==i_cluster]
        cluster_scores = [scores[i] for i in cluster_indices]
        indices[i_cluster] = [
            cluster_indices[i] for i in np.argsort(cluster_scores)
        ]
    duration_sum = 0
    score_sum = 0
    ith_best = 0
    hit_idx = []
    while True:
        best_scores = []
        for i_cluster in range(n_clusters):
            best_scores.append(
                (-scores[indices[i_cluster][ith_best]]
                 if len(indices[i_cluster])>ith_best else 1)
            )
        broken = False
        for i_cluster in np.argsort(best_scores):
            if best_scores[i_cluster]>0:
                break
            idx = indices[i_cluster][ith_best]
            if duration_sum+durations[idx]>(total_duration*(1+duration_tolerance_ratio)):
                broken = True
                break
            duration_sum += durations[idx]
            score_sum += scores[idx]
            hit_idx.append(idx) 
        if broken or len(hit_idx)>=n_vecs:
            break
            
        ith_best += 1
    kmeans_hit_pattern = [(1 if i in hit_idx else 0) for i in range(n_vecs)]

    kmeans_selection_time = time.time() - start_kmeans_selection
    print(f"score_sum = {score_sum}")
    print(f"duration_sum = {duration_sum}")
    print(f"hit_idx = {hit_idx}")
    print(f"time = {kmeans_selection_time} s")
    print(kmeans_hit_pattern)
    div_score = get_cos_sum(kmeans_hit_pattern, all_pair_cos)
    print(f"div_score = {div_score}")
    for alpha in alphas:
        L = score_sum - alpha*div_score
        print(f"L = {L} (alphas={alpha})")
    
    print("---------- all attack")
    n_all_patterns = np.power(2, n_vecs)
    all_Ls = {str(alpha):[] for alpha in alphas}
    best_L = {str(alpha):0 for alpha in alphas}
    best_pattern = {str(alpha):[] for alpha in alphas}
    best_duration_sum = {str(alpha):0 for alpha in alphas}
    start_all_attack = time.time()
    for i in range(n_all_patterns):
        hit_pattern = [int(b) for j, b in enumerate(bin(i)) if j>1]
        if len(hit_pattern)<n_vecs:
            hit_pattern += [0]*(n_vecs-len(hit_pattern))
        score_sum = 0
        this_duration_sum = 0
        for j, is_hit in enumerate(hit_pattern):
            if is_hit==0:
                continue
            score_sum += scores[j]
            this_duration_sum += durations[j]
        if np.abs(this_duration_sum/total_duration-1)>duration_tolerance_ratio:
            continue
        this_div_score = get_cos_sum(hit_pattern,all_pair_cos)

        for alpha in alphas:
            L = score_sum - alpha*this_div_score
            a_str = str(alpha)
            all_Ls[a_str].append(L)
            if best_pattern[a_str] is None or best_L[a_str]<L:
                best_L[a_str] = L
                best_pattern[a_str] = hit_pattern
                best_duration_sum[a_str] = this_duration_sum
    all_attack_time = time.time() - start_all_attack
    for alpha in alphas:
        a_str = str(alpha)
        hit_pattern = best_pattern[a_str]
        best_div_score = alpha*get_cos_sum(hit_pattern,all_pair_cos)
        best_raw_score = best_L[a_str] + best_div_score
        n_Ls = len(all_Ls[a_str])
        L_min = np.min(all_Ls[a_str])
        L_max = np.max(all_Ls[a_str])
        L_mean = np.mean(all_Ls[a_str])
        percentiles = np.percentile(all_Ls[a_str], percentile_list) 
        print(f"alphas = {a_str}")
        print(f"best_score (L) = {best_L[a_str]}")
        print(f"best_pattern = {best_pattern[a_str]}")
        print(f"best raw score = {best_raw_score}")
        print(f"duration_sum = {best_duration_sum[a_str]}")
        print(f"all Ls : len={n_Ls}, min : {L_min}, max : {L_max}, mean : {L_mean}")
        print(f"percentile: {percentiles} ({percentile_list})")
        
    print("-----")
    print(f"all_attack_time = {all_attack_time} s")
    
    full_info = {("a")+k.replace(".","_"):np.array(v)
                  for k, v in all_Ls.items()}
    full_info["scores"] = scores
    full_info["durations"] = durations
    full_info["all_pair_cos"] = all_pair_cos
    full_info["pred"] = pred
    
    np.savez(
        "comp.npz",
        **full_info,
    )

    for k, v in np.load("comp.npz").items():
        print(k, v.shape)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "compare process time between k-means clustering and "
            "cosine similarity calculation among all pairs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "n_vecs",
        type=int,
        help="# of vectors",
    )
    parser.add_argument(
        "-d", "--n_dim",
        type=int,
        default=def_n_dim,
        help="# of vector dimensions",
    )
    parser.add_argument(
        "-n", "--nominal_duration",
        type=float,
        default=def_nominal_duration,
        help="mean duration for each element",
    )
    parser.add_argument(
        "-t", "--total_duration",
        type=float,
        default=def_total_duration,
        help="total duration",
    )
    parser.add_argument(
        "-c", "--n_clusters",
        type=int,
        default=def_n_clusters,
        help="# of clusters in KMeans",
    )
    parser.add_argument(
        "-a", "--alphas",
        type=float,
        default=def_alphas,
        help="alphas",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=def_seed,
        help="random number seed",
    )

    args = parser.parse_args()
    main(
        n_vecs=args.n_vecs,
        n_dim=args.n_dim,
        nominal_duration=args.nominal_duration,
        total_duration=args.total_duration,
        n_clusters=args.n_clusters,
        alphas=args.alphas,
        seed=args.seed,
    )
