# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_MAP = {"J1": 0, "J2": 1, "J3": 2}
CLASS_ORDER = ["J1", "J2", "J3"]

def load_txt_xyzrgb(p: Path):
    arr = np.loadtxt(str(p), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[:, :6]
    xyz = arr[:, :3]
    rgb = arr[:, 3:6]
    return xyz, rgb

def pca_features(xyz: np.ndarray):
    if len(xyz) < 3:
        return {
            "eig1": 0.0, "eig2": 0.0, "eig3": 0.0,
            "nx": 0.0, "ny": 0.0, "nz": 0.0,
            "planarity": 0.0, "linearity": 0.0, "scattering": 0.0
        }

    c = xyz.mean(axis=0)
    X = xyz - c
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals = np.maximum(eigvals, 1e-12)

    # descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    e1, e2, e3 = eigvals
    normal = eigvecs[:, 2]  # smallest eigenvalue direction
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    linearity = (e1 - e2) / e1 if e1 > 0 else 0.0
    planarity = (e2 - e3) / e1 if e1 > 0 else 0.0
    scattering = e3 / e1 if e1 > 0 else 0.0

    return {
        "eig1": float(e1),
        "eig2": float(e2),
        "eig3": float(e3),
        "nx": float(normal[0]),
        "ny": float(normal[1]),
        "nz": float(normal[2]),
        "planarity": float(planarity),
        "linearity": float(linearity),
        "scattering": float(scattering),
    }

def bbox_features(xyz: np.ndarray):
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    ext = mx - mn
    return {
        "cx": float(xyz[:, 0].mean()),
        "cy": float(xyz[:, 1].mean()),
        "cz": float(xyz[:, 2].mean()),
        "bx": float(ext[0]),
        "by": float(ext[1]),
        "bz": float(ext[2]),
        "diag": float(np.linalg.norm(ext))
    }

def class_feature_vector(df_row):
    # 用“粗特征”看类间是否天然重叠
    return np.array([
        abs(df_row["nx"]),
        abs(df_row["ny"]),
        abs(df_row["nz"]),
        df_row["planarity"],
        math.log1p(df_row["n_points"]),
        df_row["mean_r"] / 255.0,
        df_row["mean_g"] / 255.0,
        df_row["mean_b"] / 255.0,
        math.log1p(df_row["diag"]),
    ], dtype=np.float64)

def nearest_centroid_confusion(df):
    feats = np.vstack([class_feature_vector(r) for _, r in df.iterrows()])
    labels = df["class_name"].values

    centroids = {}
    for cname in CLASS_ORDER:
        mask = labels == cname
        centroids[cname] = feats[mask].mean(axis=0)

    conf = pd.DataFrame(
        0,
        index=[f"GT_{c}" for c in CLASS_ORDER],
        columns=[f"Pred_{c}" for c in CLASS_ORDER],
        dtype=int
    )

    # leave-one-out nearest centroid
    for i, (_, row) in enumerate(df.iterrows()):
        gt = row["class_name"]
        x = class_feature_vector(row)

        # 对当前样本所属类做留一
        local_centroids = {}
        for cname in CLASS_ORDER:
            mask = labels == cname
            Xc = feats[mask]
            if cname == gt and Xc.shape[0] > 1:
                idxs = np.where(mask)[0]
                loc = np.where(idxs == i)[0]
                if len(loc) == 1:
                    Xc = np.delete(Xc, loc[0], axis=0)
            if Xc.shape[0] == 0:
                local_centroids[cname] = centroids[cname]
            else:
                local_centroids[cname] = Xc.mean(axis=0)

        dists = {cname: np.linalg.norm(x - local_centroids[cname]) for cname in CLASS_ORDER}
        pred = min(dists, key=dists.get)
        conf.loc[f"GT_{gt}", f"Pred_{pred}"] += 1

    return conf

def save_gallery(df, out_png, max_show_per_class=6, max_points_show=4000):
    fig = plt.figure(figsize=(3.2 * max_show_per_class, 9.5))
    plot_idx = 1

    for row_i, cname in enumerate(CLASS_ORDER):
        sub = df[df["class_name"] == cname].copy()
        # 取若干个中等/代表性实例，不只看最大最小
        sub = sub.sort_values("n_points").reset_index(drop=True)
        if len(sub) == 0:
            continue
        pick_idx = np.linspace(0, len(sub) - 1, num=min(max_show_per_class, len(sub)), dtype=int)

        for j in pick_idx:
            r = sub.iloc[j]
            xyz, rgb = load_txt_xyzrgb(Path(r["file_path"]))
            if len(xyz) > max_points_show:
                inds = np.random.choice(len(xyz), max_points_show, replace=False)
                xyz = xyz[inds]
                rgb = rgb[inds]

            ax = fig.add_subplot(len(CLASS_ORDER), max_show_per_class, plot_idx, projection='3d')
            plot_idx += 1

            rgb_show = np.clip(rgb / 255.0, 0, 1)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb_show, s=1)
            ax.set_title(f"{cname}\n{Path(r['file_name']).stem}\nN={int(r['n_points'])}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(ann_dir.glob("*.txt")):
        stem = p.stem
        if stem.startswith("merged_"):
            continue
        cls = stem.split("_")[0]
        if cls not in CLASS_MAP:
            continue

        xyz, rgb = load_txt_xyzrgb(p)
        pca = pca_features(xyz)
        bbox = bbox_features(xyz)

        row = {
            "file_name": p.name,
            "file_path": str(p),
            "class_name": cls,
            "class_id": CLASS_MAP[cls],
            "n_points": int(len(xyz)),
            "mean_r": float(rgb[:, 0].mean()),
            "mean_g": float(rgb[:, 1].mean()),
            "mean_b": float(rgb[:, 2].mean()),
        }
        row.update(bbox)
        row.update(pca)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"在 {ann_dir} 没找到 J1/J2/J3 实例文件。")

    # 保存逐实例统计
    inst_csv = out_dir / "instance_stats.csv"
    df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    # 每类汇总
    summary = df.groupby("class_name").agg(
        num_instances=("file_name", "count"),
        total_points=("n_points", "sum"),
        mean_points=("n_points", "mean"),
        median_points=("n_points", "median"),
        q25_points=("n_points", lambda x: np.percentile(x, 25)),
        q75_points=("n_points", lambda x: np.percentile(x, 75)),
        mean_planarity=("planarity", "mean"),
        median_planarity=("planarity", "median"),
        mean_r=("mean_r", "mean"),
        mean_g=("mean_g", "mean"),
        mean_b=("mean_b", "mean"),
        mean_abs_nx=("nx", lambda x: np.mean(np.abs(x))),
        mean_abs_ny=("ny", lambda x: np.mean(np.abs(x))),
        mean_abs_nz=("nz", lambda x: np.mean(np.abs(x))),
    ).reset_index()
    summary_csv = out_dir / "class_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 粗特征最近类中心混淆
    conf = nearest_centroid_confusion(df)
    conf_csv = out_dir / "nearest_centroid_confusion.csv"
    conf.to_csv(conf_csv, encoding="utf-8-sig")

    # 可视化拼图
    gallery_png = out_dir / "instance_gallery.png"
    save_gallery(df, gallery_png)

    print("=" * 100)
    print("saved:", inst_csv)
    print("saved:", summary_csv)
    print("saved:", conf_csv)
    print("saved:", gallery_png)
    print("=" * 100)
    print("Class summary:")
    print(summary.to_string(index=False))
    print("=" * 100)
    print("Nearest-centroid confusion (instance-level coarse features):")
    print(conf.to_string())
    print("=" * 100)

if __name__ == "__main__":
    main()
