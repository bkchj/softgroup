import os
import torch
import numpy as np
from pathlib import Path

SRC_ROOT = Path('/home/chj/SoftGroup/dataset/rockjoint_split_j1j5_v1')
DST_ROOT = Path('/home/chj/SoftGroup/dataset/rockjoint_split_j1j5_xyzonly_v2')

SPLITS = ['train', 'val', 'test_a', 'test_b']

# 清空旧目录
if DST_ROOT.exists():
    import shutil
    shutil.rmtree(DST_ROOT)

for split in SPLITS:
    (DST_ROOT / split).mkdir(parents=True, exist_ok=True)

print("开始构建 xyz-only 数据集...")

for split in SPLITS:
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split

    files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pth')])

    print(f"\n[{split}] 共 {len(files)} 个文件")

    for i, fn in enumerate(files, 1):
        src_path = src_dir / fn
        dst_path = dst_dir / fn

        xyz, feat_old, sem, ins = torch.load(src_path, map_location='cpu', weights_only=False)

        xyz = np.asarray(xyz, dtype=np.float32)
        sem = np.asarray(sem, dtype=np.int64)
        ins = np.asarray(ins, dtype=np.int64)

        # 🚩核心：feat = 全零
        feat = np.zeros_like(xyz, dtype=np.float32)

        torch.save((xyz, feat, sem, ins), dst_path)

        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

print("\n✅ xyz-only 数据集构建完成")
print(f"输出路径: {DST_ROOT}")
