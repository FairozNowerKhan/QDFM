import numpy as np

def _kmeans_np(X, K, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, size=K, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        C_new = np.zeros_like(C)
        for k in range(K):
            mask = (a == k)
            if mask.any():
                C_new[k] = X[mask].mean(0)
            else:
                C_new[k] = X[rng.integers(0, N)]
        if np.max(np.abs(C_new - C)) < 1e-6:
            C = C_new
            break
        C = C_new
    return C


def build_centroids(actions, K, seed, iters=40):
    actions = np.asarray(actions, dtype=np.float32)
    N = actions.shape[0]
    if N > 250000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=250000, replace=False)
        actions = actions[idx]
    C = _kmeans_np(actions, K=K, iters=iters, seed=seed)
    return C.astype(np.float32)


def assign_to_centroids(actions, centroids):
    A = np.asarray(actions, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)
    d2 = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1).astype(np.int64)
