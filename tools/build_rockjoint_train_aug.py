# -*- coding: utf-8 -*-
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch

# =========================
# 路径配置
# =========================
SRC_ROOT = Path("/home/chj/SoftGroup/dataset/rockjoint_split")
DST_ROOT = Path("/home/chj/SoftGroup/dataset/rockjoint_split_aug_v2")

TRAIN_FILE = SRC_ROOT / "train" / "wysbp_train.pth"
VAL_FILE = SRC_ROOT / "val" / "wysbp_val.pth"
TEST_A_FILE = SRC_ROOT / "test" / "wysbp_test_a.pth"
TEST_B_FILE = SRC_ROOT / "test" / "wysbp_test_b.pth"

# class 0 = J1, class 1 = J2, class 2 = J3
CLASS_NAMES = {0: "J1", 1: "J2", 2: "J3"}

# 只生成 2 个增强版本，避免增强样本占比过大
NUM_AUG_FILES = 2
BASE_SEED = 20260410

# 阈值参考你当前真实统计
SIZE_THRESHOLDS = {
    0: {"small": 250, "mid": 700, "large": 1600},   # J1
    1: {"small": 300, "mid": 1000, "large": 2300},  # J2
    2: {"small": 400, "mid": 1500, "large": 3500},  # J3
}


def sample_keep_ratio(cls_id: int, inst_size: int, rng: np.random.Generator) -> float:
    """
    温和版 v2：
    - J1 仍然增强更强，但不再像 v1 那样压得太狠
    - J2 中等
    - J3 只做轻度削弱密度优势
    """
    th = SIZE_THRESHOLDS[cls_id]

    if cls_id == 0:  # J1
        if inst_size < th["small"]:
            return 1.0
        elif inst_size < th["mid"]:
            return float(rng.uniform(0.80, 0.95))
        elif inst_size < th["large"]:
            return float(rng.uniform(0.60, 0.85))
        else:
            return float(rng.uniform(0.45, 0.75))

    if cls_id == 1:  # J2
        if inst_size < th["small"]:
            return 1.0
        elif inst_size < th["mid"]:
            return float(rng.uniform(0.85, 0.95))
        elif inst_size < th["large"]:
            return float(rng.uniform(0.70, 0.88))
        else:
            return float(rng.uniform(0.55, 0.80))

    if cls_id == 2:  # J3
        if inst_size < th["small"]:
            return 1.0
        elif inst_size < th["mid"]:
            return float(rng.uniform(0.90, 0.97))
        elif inst_size < th["large"]:
            return float(rng.uniform(0.80, 0.92))
        else:
            return float(rng.uniform(0.65, 0.85))

    return 1.0


def load_tuple_pth(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(data, tuple) or len(data) != 4:
        raise ValueError(f"{p} is not a 4-tuple .pth file")
    xyz, feats, sem, ins = data
    return np.asarray(xyz), np.asarray(feats), np.asarray(sem), np.asarray(ins)


def save_tuple_pth(p: Path, xyz: np.ndarray, feats: np.ndarray, sem: np.ndarray, ins: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save((xyz.astype(np.float32),
                feats.astype(np.float32),
                sem.astype(np.int64),
                ins.astype(np.int64)), p)


def relabel_instances_sequential(ins: np.ndarray) -> np.ndarray:
    uniq = np.unique(ins)
    uniq = uniq[uniq >= 0]
    mapping = {old: new for new, old in enumerate(uniq)}
    out = np.full_like(ins, fill_value=-1)
    for old, new in mapping.items():
        out[ins == old] = new
    return out


def majority_class(sem_vals: np.ndarray) -> int:
    uniq, cnt = np.unique(sem_vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def augment_train_once(
    xyz: np.ndarray,
    feats: np.ndarray,
    sem: np.ndarray,
    ins: np.ndarray,
    seed: int,
):
    rng = np.random.default_rng(seed)
    keep_mask = np.zeros(len(sem), dtype=bool)

    uniq_ins = np.unique(ins)
    uniq_ins = uniq_ins[uniq_ins >= 0]

    stats = {
        "seed": seed,
        "before_points": int(len(sem)),
        "after_points": 0,
        "class_before_points": {},
        "class_after_points": {},
        "class_num_instances": {},
        "class_keep_ratio_mean": {},
    }

    per_class_keep_ratios: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    for inst_id in uniq_ins:
        idx = np.where(ins == inst_id)[0]
        if idx.size == 0:
            continue

        cls_id = majority_class(sem[idx])
        inst_size = idx.size

        keep_ratio = sample_keep_ratio(cls_id, inst_size, rng)
        per_class_keep_ratios[cls_id].append(keep_ratio)

        if keep_ratio >= 0.999:
            chosen = idx
        else:
            n_keep = max(2, int(round(inst_size * keep_ratio)))
            n_keep = min(n_keep, inst_size)
            chosen = rng.choice(idx, size=n_keep, replace=False)

        keep_mask[chosen] = True

    orphan = (ins < 0)
    if np.any(orphan):
        keep_mask[orphan] = True

    xyz_new = xyz[keep_mask]
    feats_new = feats[keep_mask]
    sem_new = sem[keep_mask]
    ins_new = ins[keep_mask]
    ins_new = relabel_instances_sequential(ins_new)

    stats["after_points"] = int(len(sem_new))
    for c in [0, 1, 2]:
        stats["class_before_points"][c] = int((sem == c).sum())
        stats["class_after_points"][c] = int((sem_new == c).sum())
        stats["class_num_instances"][c] = int(sum(1 for x in uniq_ins if majority_class(sem[ins == x]) == c))
        stats["class_keep_ratio_mean"][c] = (
            float(np.mean(per_class_keep_ratios[c])) if len(per_class_keep_ratios[c]) > 0 else 1.0
        )

    return xyz_new, feats_new, sem_new, ins_new, stats


def print_stats_table(title: str, sem: np.ndarray, ins: np.ndarray) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"total_points = {len(sem)}")

    uniq_sem, cnt_sem = np.unique(sem, return_counts=True)
    print("\n[Point counts by class]")
    for c, cnt in zip(uniq_sem, cnt_sem):
        ratio = cnt / len(sem)
        name = CLASS_NAMES.get(int(c), str(int(c)))
        print(f"class {int(c)} ({name}): points={cnt}, ratio={ratio:.4f}")

    print("\n[Instance stats by class]")
    for c in sorted(np.unique(sem)):
        inst_sizes = []
        uniq_ins = np.unique(ins[sem == c])
        uniq_ins = uniq_ins[uniq_ins >= 0]
        for inst_id in uniq_ins:
            size = int((ins == inst_id).sum())
            if size > 0:
                inst_sizes.append(size)
        if len(inst_sizes) == 0:
            continue
        arr = np.array(inst_sizes, dtype=np.int64)
        name = CLASS_NAMES.get(int(c), str(int(c)))
        print(
            f"class {int(c)} ({name}): "
            f"num_instances={len(arr)}, "
            f"mean={arr.mean():.2f}, "
            f"median={np.median(arr):.2f}, "
            f"min={arr.min()}, "
            f"p25={np.percentile(arr,25):.2f}, "
            f"p75={np.percentile(arr,75):.2f}, "
            f"max={arr.max()}"
        )


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    for sub in ["train", "val", "test"]:
        (DST_ROOT / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(VAL_FILE, DST_ROOT / "val" / VAL_FILE.name)
    shutil.copy2(TEST_A_FILE, DST_ROOT / "test" / TEST_A_FILE.name)
    shutil.copy2(TEST_B_FILE, DST_ROOT / "test" / TEST_B_FILE.name)

    xyz, feats, sem, ins = load_tuple_pth(TRAIN_FILE)
    print_stats_table("ORIGINAL TRAIN", sem, ins)

    save_tuple_pth(DST_ROOT / "train" / "wysbp_train_orig.pth", xyz, feats, sem, ins)

    all_stats = []
    for i in range(NUM_AUG_FILES):
        seed = BASE_SEED + i
        xyz_new, feats_new, sem_new, ins_new, stats = augment_train_once(xyz, feats, sem, ins, seed)
        out_path = DST_ROOT / "train" / f"wysbp_train_aug_{i+1}.pth"
        save_tuple_pth(out_path, xyz_new, feats_new, sem_new, ins_new)

        print_stats_table(f"AUG TRAIN {i+1}", sem_new, ins_new)
        print(f"\nSaved to: {out_path}")
        print(f"Keep ratio mean by class: {stats['class_keep_ratio_mean']}")
        all_stats.append((out_path.name, stats))

    summary_path = DST_ROOT / "train" / "augmentation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for name, st in all_stats:
            f.write(f"{name}\n")
            f.write(f"  seed: {st['seed']}\n")
            f.write(f"  before_points: {st['before_points']}\n")
            f.write(f"  after_points: {st['after_points']}\n")
            for c in [0, 1, 2]:
                f.write(
                    f"  class {c} ({CLASS_NAMES[c]}): "
                    f"before_points={st['class_before_points'][c]}, "
                    f"after_points={st['class_after_points'][c]}, "
                    f"num_instances={st['class_num_instances'][c]}, "
                    f"mean_keep_ratio={st['class_keep_ratio_mean'][c]:.4f}\n"
                )
            f.write("\n")

    print("\nDone.")
    print(f"Augmented dataset saved to: {DST_ROOT}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
