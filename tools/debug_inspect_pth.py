# tools/debug_inspect_pth.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_numpy(x: Any):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None


def describe_array(name: str, x: Any):
    arr = to_numpy(x)
    if arr is None:
        print(f"\n{name}: type={type(x)}")
        return

    print(f"\n{name}:")
    print(f"  shape = {arr.shape}")
    print(f"  dtype = {arr.dtype}")

    if arr.size == 0:
        print("  empty array")
        return

    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1)
        print(f"  min   = {flat.min()}")
        print(f"  max   = {flat.max()}")
        print(f"  mean  = {flat.mean()}")
        print(f"  std   = {flat.std()}")

    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        vals, counts = np.unique(arr, return_counts=True)
        pairs = list(zip(vals.tolist(), counts.tolist()))
        print(f"  unique/counts = {pairs[:50]}")
        total = counts.sum()
        if total > 0:
            print(
                "  ratio = "
                + str([(int(v), round(float(c) / float(total), 6)) for v, c in pairs[:50]])
            )


def inspect_obj(obj: Any, prefix: str = "root"):
    if isinstance(obj, dict):
        print(f"{prefix}: dict keys = {list(obj.keys())}")
        for k, v in obj.items():
            inspect_obj(v, f"{prefix}.{k}")
        return

    if isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__}, len = {len(obj)}")
        for i, v in enumerate(obj):
            inspect_obj(v, f"{prefix}[{i}]")
        return

    describe_array(prefix, obj)


def safe_torch_load(path: Path):
    """
    兼容 PyTorch 2.6+ 默认 weights_only=True 的变化。
    对于数据样本 .pth，需要显式使用 weights_only=False。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版 PyTorch（没有 weights_only 参数）
        return torch.load(path, map_location="cpu")


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/debug_inspect_pth.py <path_to_pth>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    print(f"Loading: {p}")
    obj = safe_torch_load(p)
    inspect_obj(obj)


if __name__ == "__main__":
    main()
