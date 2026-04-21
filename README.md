# SoftGroup for Rock Joint Segmentation (J1–J5)

## 1. Overview

This repository provides a reproducible implementation of voxel-based semantic segmentation (SoftGroup) for rock joint identification.

The dataset consists of tunnel face point clouds with five discontinuity sets (J1–J5).

---

## 2. Dataset

Structure:

dataset/
└── rockjoint_split_j1j5_xyzonly_v2/
    ├── train/
    ├── val/
    └── test_a/

Data format:

Each .pth file contains:
(xyz, feat, semantic_label, instance_label)

- xyz: (N, 3)
- feat: (N, 3) (all zeros)
- semantic_label: {0,1,2,3,4} → J1–J5
- instance_label: instance ids

---

## 3. Training

Best config:

configs/softgroup_rockjoint_j1j5_semantic_only_vox0p022.yaml

Run:

python tools/train.py configs/softgroup_rockjoint_j1j5_semantic_only_vox0p022.yaml

---

## 4. Results

Best:

mIoU = 73.1  
Acc  = 93.0  

Class-wise IoU:

J1 = 95.1  
J2 = 35.7  
J3 = 93.5  
J4 = 77.6  
J5 = 63.7  

---

## 5. Key Findings

- Optimal voxel size ≈ 0.02–0.022 m  
- XYZ-only already works for major structures  
- Voxel-based > point-based > query-based  

---

## 6. Notes

- No RGB / normals used  
- Instance segmentation not used in final  

