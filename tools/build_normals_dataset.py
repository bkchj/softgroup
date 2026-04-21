import os
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

SRC_ROOT = Path('/home/chj/SoftGroup/dataset/rockjoint_split_j1j5_v1')
DST_ROOT = Path('/home/chj/SoftGroup/dataset/rockjoint_split_j1j5_normal_v1')
SPLITS = ['train', 'val', 'test_a', 'test_b']

# k近邻数，建议先用20~30
K = 24

def estimate_normals_pca(xyz: np.ndarray, k: int = 24) -> np.ndarray:
    """
    xyz: (N,3) float32
    return normals: (N,3) float32, unit normals in [-1,1]
    """
    n = xyz.shape[0]
    if n < 4:
        # 点太少时无法稳定估计，返回零法向
        return np.zeros((n, 3), dtype=np.float32)

    k_use = min(k, max(3, n - 1))

    nbrs = NearestNeighbors(n_neighbors=k_use + 1, algorithm='auto')
    nbrs.fit(xyz)
    indices = nbrs.kneighbors(xyz, return_distance=False)

    normals = np.zeros((n, 3), dtype=np.float32)

    center = xyz.mean(axis=0)

    for i in range(n):
        # 第一个邻居是自己，去掉
        nn_idx = indices[i, 1:]
        pts = xyz[nn_idx]

        if pts.shape[0] < 3:
            continue

        mu = pts.mean(axis=0, keepdims=True)
        X = pts - mu
        cov = (X.T @ X) / max(1, X.shape[0])

        # 最小特征值对应法向
        eigvals, eigvecs = np.linalg.eigh(cov)
        nvec = eigvecs[:, 0]

        # 方向一致化：让法向尽量朝向远离整体中心的方向
        # 避免同一块内正负翻转过多
        ref = xyz[i] - center
        if np.dot(nvec, ref) < 0:
            nvec = -nvec

        norm = np.linalg.norm(nvec)
        if norm > 1e-12:
            nvec = nvec / norm
        else:
            nvec = np.zeros(3, dtype=np.float32)

        normals[i] = nvec.astype(np.float32)

    return normals.astype(np.float32)

def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f'Source dataset not found: {SRC_ROOT}')

    if DST_ROOT.exists():
        import shutil
        shutil.rmtree(DST_ROOT)

    for split in SPLITS:
        (DST_ROOT / split).mkdir(parents=True, exist_ok=True)

    print(f'SRC_ROOT = {SRC_ROOT}')
    print(f'DST_ROOT = {DST_ROOT}')
    print(f'K = {K}')

    for split in SPLITS:
        src_dir = SRC_ROOT / split
        dst_dir = DST_ROOT / split

        files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pth')])
        print(f'\n[{split}] files = {len(files)}')

        for i, fn in enumerate(files, 1):
            src_path = src_dir / fn
            dst_path = dst_dir / fn

            xyz, feat_old, sem, ins = torch.load(src_path, map_location='cpu', weights_only=False)

            xyz = np.asarray(xyz, dtype=np.float32)
            sem = np.asarray(sem, dtype=np.int64)
            ins = np.asarray(ins, dtype=np.int64)

            normals = estimate_normals_pca(xyz, k=K)

            # 保存为 (xyz, feat, semantic, instance)
            # 这里 feat = normals
            torch.save((xyz, normals, sem, ins), dst_path)

            if i == 1 or i % 10 == 0 or i == len(files):
                u_sem = np.unique(sem).tolist()
                print(f'  {i:>3}/{len(files)}  {fn}  points={len(xyz)}  classes={u_sem}')

    print('\nDone.')
    print(f'Output dataset: {DST_ROOT}')

if __name__ == '__main__':
    main()
