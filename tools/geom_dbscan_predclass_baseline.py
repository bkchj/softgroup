# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


DEFAULT_DATA_ROOT = Path("/home/chj/SoftGroup/dataset/rockjoint_split")
DEFAULT_PRED_ROOT = Path("/home/chj/PointBaselineCompare/outputs")


def load_gt_split(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(data, tuple) and len(data) == 4):
        raise ValueError(f"{path} is not a 4-tuple dataset file")
    xyz, feats, sem, ins = data
    xyz = np.asarray(xyz, dtype=np.float32)
    sem = np.asarray(sem, dtype=np.int64)
    ins = np.asarray(ins, dtype=np.int64)
    return xyz, sem, ins


def load_pred_npz(path: Path):
    d = np.load(path, allow_pickle=True)

    xyz = np.asarray(d["points"][:, :3], dtype=np.float32) if "points" in d.files else None
    gt_sem = np.asarray(d["gt_sem"], dtype=np.int64)
    pred_sem = np.asarray(d["pred_sem"], dtype=np.int64)

    if "choice" not in d.files:
        raise ValueError(f"{path} missing required key: 'choice'")

    choice = np.asarray(d["choice"], dtype=np.int64)

    return xyz, gt_sem, pred_sem, choice


def unique_valid_instances(sem: np.ndarray, ins: np.ndarray, class_id: int):
    masks = []
    valid = (sem == class_id) & (ins >= 0)
    if valid.sum() == 0:
        return masks
    inst_ids = np.unique(ins[valid])
    for iid in inst_ids:
        m = valid & (ins == iid)
        if m.sum() > 0:
            masks.append(m)
    return masks


def kth_neighbor_scale(points: np.ndarray, k: int) -> float:
    n = points.shape[0]
    if n <= 2:
        return 1.0
    kk = min(max(k, 2), n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto", n_jobs=-1)
    nn.fit(points)
    dists, _ = nn.kneighbors(points, return_distance=True)
    kd = dists[:, -1]
    scale = float(np.median(kd))
    if not np.isfinite(scale) or scale <= 1e-8:
        pos = kd[kd > 0]
        scale = float(np.mean(pos)) if len(pos) > 0 else 1.0
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return scale


def dbscan_cluster_xyz(xyz: np.ndarray, class_mask: np.ndarray, alpha: float, min_samples: int):
    idx = np.where(class_mask)[0]
    if idx.size == 0:
        return []

    pts = xyz[idx]
    if pts.shape[0] < min_samples:
        return []

    scale = kth_neighbor_scale(pts, k=min_samples)
    eps = max(alpha * scale, 1e-6)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = clustering.fit_predict(pts)

    pred_masks = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        local = (labels == cid)
        if local.sum() == 0:
            continue
        full_mask = np.zeros(xyz.shape[0], dtype=bool)
        full_mask[idx[local]] = True
        pred_masks.append(full_mask)
    return pred_masks


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def precision_recall_ap(preds, gts, iou_thr: float):
    if len(preds) == 0:
        n_gt = sum(len(v) for v in gts.values())
        return 0.0, 0.0, 0.0 if n_gt > 0 else (0.0, 0.0, 0.0)

    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

    matched = {}
    for key, masks in gts.items():
        matched[key] = np.zeros(len(masks), dtype=bool)

    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)
    n_gt_total = sum(len(v) for v in gts.values())

    for i, pred in enumerate(preds):
        key = (pred["scene_id"], pred["class_id"])
        gt_masks = gts.get(key, [])
        if len(gt_masks) == 0:
            fp[i] = 1
            continue

        best_iou = -1.0
        best_j = -1
        for j, gt_mask in enumerate(gt_masks):
            if matched[key][j]:
                continue
            v = iou_binary(pred["mask"], gt_mask)
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1
            matched[key][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(n_gt_total, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1])

    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    return float(ap), float(final_recall), float(final_precision)


def eval_one_split(split_name, xyz, gt_sem, gt_ins, pred_sem, params_by_class, class_names):
    preds_all = {0: [], 1: [], 2: []}
    gts_all = {0: {}, 1: {}, 2: {}}
    scene_id = split_name

    for class_id in [0, 1, 2]:
        gt_masks = unique_valid_instances(gt_sem, gt_ins, class_id)
        gts_all[class_id][(scene_id, class_id)] = gt_masks

        class_mask = (pred_sem == class_id)
        pred_masks = dbscan_cluster_xyz(
            xyz=xyz,
            class_mask=class_mask,
            alpha=params_by_class[class_id]["alpha"],
            min_samples=int(params_by_class[class_id]["min_samples"]),
        )

        for pm in pred_masks:
            preds_all[class_id].append(
                {
                    "scene_id": scene_id,
                    "class_id": class_id,
                    "conf": float(pm.sum()),
                    "mask": pm,
                }
            )

    thresholds = [0.50 + 0.05 * i for i in range(10)]
    results = {}

    for class_id in [0, 1, 2]:
        ap_list = []
        ap50, ap25 = 0.0, 0.0
        ar50, ar25 = 0.0, 0.0

        for thr in thresholds:
            ap, recall, _ = precision_recall_ap(preds_all[class_id], gts_all[class_id], thr)
            ap_list.append(ap)
            if abs(thr - 0.50) < 1e-9:
                ap50 = ap
                ar50 = recall

        ap25, ar25, _ = precision_recall_ap(preds_all[class_id], gts_all[class_id], 0.25)
        mean_ap = float(np.mean(ap_list)) if len(ap_list) else 0.0
        mean_ar = float(np.mean([precision_recall_ap(preds_all[class_id], gts_all[class_id], thr)[1] for thr in thresholds]))

        results[class_id] = {
            "name": class_names[class_id],
            "AP": mean_ap,
            "AP_50": ap50,
            "AP_25": ap25,
            "AR": mean_ar,
            "RC_50": ar50,
            "RC_25": ar25,
            "num_pred": len(preds_all[class_id]),
            "num_gt": len(gts_all[class_id][(scene_id, class_id)]),
        }

    return results


def merge_split_results_for_report(split_results: Dict[int, Dict[str, float]]):
    avg = {}
    keys = ["AP", "AP_50", "AP_25", "AR", "RC_50", "RC_25"]
    for k in keys:
        avg[k] = float(np.mean([split_results[c][k] for c in [0, 1, 2]]))
    return avg


def pretty_print(split_name: str, results: Dict[int, Dict[str, float]]):
    avg = merge_split_results_for_report(results)
    print("\n" + "=" * 72)
    print(f"SPLIT: {split_name}")
    print("#" * 64)
    print(f"{'what':<12}{'AP':>10}{'AP_50%':>10}{'AP_25%':>10}{'AR':>10}{'RC_50%':>10}{'RC_25%':>10}")
    print("#" * 64)
    for cid in [0, 1, 2]:
        r = results[cid]
        print(f"{r['name']:<12}{r['AP']:>10.3f}{r['AP_50']:>10.3f}{r['AP_25']:>10.3f}{r['AR']:>10.3f}{r['RC_50']:>10.3f}{r['RC_25']:>10.3f}")
    print("-" * 64)
    print(f"{'average':<12}{avg['AP']:>10.3f}{avg['AP_50']:>10.3f}{avg['AP_25']:>10.3f}{avg['AR']:>10.3f}{avg['RC_50']:>10.3f}{avg['RC_25']:>10.3f}")
    print("#" * 64)


def tune_on_val(xyz, gt_sem, gt_ins, pred_sem, class_names):
    alpha_grid = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    min_samples_grid = [4, 6, 8, 10, 12, 15]
    best = {}

    for class_id in [0, 1, 2]:
        best_score = -1.0
        best_cfg = None

        gt_masks = unique_valid_instances(gt_sem, gt_ins, class_id)
        if len(gt_masks) == 0:
            best[class_id] = {"alpha": 2.0, "min_samples": 8}
            continue

        for alpha, min_samples in itertools.product(alpha_grid, min_samples_grid):
            class_mask = (pred_sem == class_id)
            pred_masks = dbscan_cluster_xyz(xyz, class_mask, alpha=alpha, min_samples=min_samples)

            preds = [
                {"scene_id": "val", "class_id": class_id, "conf": float(pm.sum()), "mask": pm}
                for pm in pred_masks
            ]
            gts = {("val", class_id): gt_masks}

            ap_mean_list = []
            for thr in [0.50 + 0.05 * i for i in range(10)]:
                ap_thr, _, _ = precision_recall_ap(preds, gts, thr)
                ap_mean_list.append(ap_thr)

            ap = float(np.mean(ap_mean_list))
            ap50, _, _ = precision_recall_ap(preds, gts, 0.50)
            ap25, _, _ = precision_recall_ap(preds, gts, 0.25)

            score = ap + 0.25 * ap50 + 0.10 * ap25
            if score > best_score:
                best_score = score
                best_cfg = {"alpha": alpha, "min_samples": min_samples}

        best[class_id] = best_cfg
        print(f"[TUNE] class {class_id} ({class_names[class_id]}): best={best_cfg}, score={best_score:.4f}")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pred_root", type=str, default=str(DEFAULT_PRED_ROOT))
    parser.add_argument("--out_txt", type=str, default="work_dirs/geom_dbscan_predclass_report.txt")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    class_names = {0: "J1", 1: "J2", 2: "J3"}

    # -----------------------------
    # load GT full scenes
    # -----------------------------
    xyz_val_full, gt_sem_val_full, gt_ins_val_full = load_gt_split(gt_root / "val" / "wysbp_val.pth")
    xyz_ta_full, gt_sem_ta_full, gt_ins_ta_full = load_gt_split(gt_root / "test" / "wysbp_test_a.pth")
    xyz_tb_full, gt_sem_tb_full, gt_ins_tb_full = load_gt_split(gt_root / "test" / "wysbp_test_b.pth")

    # -----------------------------
    # load PointNet++ predictions
    # -----------------------------
    _, gt_sem_pred_val, pred_sem_val, choice_val = load_pred_npz(
        pred_root / "pointnet2_semseg_rockjoint_pred_val" / "wysbp_val.npz"
    )
    _, gt_sem_pred_ta, pred_sem_ta, choice_ta = load_pred_npz(
        pred_root / "pointnet2_semseg_rockjoint_pred_test_a" / "wysbp_test_a.npz"
    )
    _, gt_sem_pred_tb, pred_sem_tb, choice_tb = load_pred_npz(
        pred_root / "pointnet2_semseg_rockjoint_pred_test_b" / "wysbp_test_b.npz"
    )

    # -----------------------------
    # exact alignment using choice
    # -----------------------------
    xyz_val = xyz_val_full[choice_val]
    gt_sem_val = gt_sem_val_full[choice_val]
    gt_ins_val = gt_ins_val_full[choice_val]

    xyz_ta = xyz_ta_full[choice_ta]
    gt_sem_ta = gt_sem_ta_full[choice_ta]
    gt_ins_ta = gt_ins_ta_full[choice_ta]

    xyz_tb = xyz_tb_full[choice_tb]
    gt_sem_tb = gt_sem_tb_full[choice_tb]
    gt_ins_tb = gt_ins_tb_full[choice_tb]

    # optional consistency check
    assert len(gt_sem_val) == len(pred_sem_val)
    assert len(gt_sem_ta) == len(pred_sem_ta)
    assert len(gt_sem_tb) == len(pred_sem_tb)

    print("[CHECK] val   aligned points:", len(pred_sem_val), "GT match rate:", float((gt_sem_val == gt_sem_pred_val).mean()))
    print("[CHECK] test_a aligned points:", len(pred_sem_ta), "GT match rate:", float((gt_sem_ta == gt_sem_pred_ta).mean()))
    print("[CHECK] test_b aligned points:", len(pred_sem_tb), "GT match rate:", float((gt_sem_tb == gt_sem_pred_tb).mean()))

    print("[INFO] tuning on val (pred-class) ...")
    params = tune_on_val(xyz_val, gt_sem_val, gt_ins_val, pred_sem_val, class_names)
    print("[INFO] tuned params:", params)

    res_val = eval_one_split("val", xyz_val, gt_sem_val, gt_ins_val, pred_sem_val, params, class_names)
    res_ta = eval_one_split("test_a", xyz_ta, gt_sem_ta, gt_ins_ta, pred_sem_ta, params, class_names)
    res_tb = eval_one_split("test_b", xyz_tb, gt_sem_tb, gt_ins_tb, pred_sem_tb, params, class_names)

    pretty_print("val", res_val)
    pretty_print("test_a", res_ta)
    pretty_print("test_b", res_tb)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("[TUNED PARAMS]\n")
        for cid in [0, 1, 2]:
            f.write(f"class {cid} ({class_names[cid]}): {params[cid]}\n")
        f.write("\n")

        for split_name, res in [("val", res_val), ("test_a", res_ta), ("test_b", res_tb)]:
            avg = merge_split_results_for_report(res)
            f.write("=" * 72 + "\n")
            f.write(f"SPLIT: {split_name}\n")
            f.write("#" * 64 + "\n")
            f.write(f"{'what':<12}{'AP':>10}{'AP_50%':>10}{'AP_25%':>10}{'AR':>10}{'RC_50%':>10}{'RC_25%':>10}\n")
            f.write("#" * 64 + "\n")
            for cid in [0, 1, 2]:
                r = res[cid]
                f.write(f"{r['name']:<12}{r['AP']:>10.3f}{r['AP_50']:>10.3f}{r['AP_25']:>10.3f}{r['AR']:>10.3f}{r['RC_50']:>10.3f}{r['RC_25']:>10.3f}\n")
            f.write("-" * 64 + "\n")
            f.write(f"{'average':<12}{avg['AP']:>10.3f}{avg['AP_50']:>10.3f}{avg['AP_25']:>10.3f}{avg['AR']:>10.3f}{avg['RC_50']:>10.3f}{avg['RC_25']:>10.3f}\n")
            f.write("#" * 64 + "\n\n")

    print(f"\n[INFO] report saved to: {out_txt}")


if __name__ == "__main__":
    main()
