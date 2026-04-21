# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from softgroup.data import build_dataloader, build_dataset
from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint
from munch import Munch
import yaml


CLASS_NAMES = {
    0: 'J1',
    1: 'J2',
    2: 'J3',
}


def dict_to_munch(obj):
    if isinstance(obj, dict):
        return Munch({k: dict_to_munch(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_munch(x) for x in obj]
    return obj


def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return dict_to_munch(cfg)


def build_model_from_cfg(cfg):
    model = SoftGroup(**cfg.model)
    return model


def safe_to_cuda(batch):
    """
    只把 tensor 搬到 cuda，字符串/list/dict 保持不动
    """
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.cuda(non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def to_numpy_int64(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1).astype(np.int64)
    if torch.is_tensor(x):
        return x.reshape(-1).detach().cpu().numpy().astype(np.int64)
    return np.asarray(x).reshape(-1).astype(np.int64)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='optional output csv path'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_root_logger()

    # 强制只做语义推理，避免实例评估路径继续干扰分析
    if not hasattr(cfg.model, 'test_cfg'):
        raise ValueError('cfg.model.test_cfg 不存在')
    cfg.model.test_cfg.eval_tasks = ['semantic']

    # 构建 val 数据集
    val_set = build_dataset(cfg.data.test, logger)
    val_loader = build_dataloader(
        val_set,
        batch_size=cfg.dataloader.test.batch_size,
        num_workers=cfg.dataloader.test.num_workers,
        training=False,
        dist=False
    )

    # 构建模型并加载 checkpoint
    model = build_model_from_cfg(cfg).cuda()
    load_checkpoint(args.checkpoint, logger, model)
    model.eval()

    num_classes = int(cfg.model.semantic_classes)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 统计每类真值总点数、预测总点数
    gt_count = np.zeros(num_classes, dtype=np.int64)
    pred_count = np.zeros(num_classes, dtype=np.int64)

    for batch in val_loader:
        batch = safe_to_cuda(batch)

        # forward_test
        result = model(batch)

        # 兼容不同返回格式
        # 常见情况：result 里有 semantic_preds
        if isinstance(result, dict):
            if 'semantic_preds' in result:
                pred_sem = result['semantic_preds']
            elif 'semantic_pred' in result:
                pred_sem = result['semantic_pred']
            else:
                raise KeyError(f'结果里没找到 semantic_preds / semantic_pred，当前 keys={list(result.keys())}')
        else:
            raise TypeError(f'不支持的 result 类型: {type(result)}')

        # gt
        gt_sem = batch['semantic_labels']

        # 展平
        pred_sem = to_numpy_int64(pred_sem)
        gt_sem = to_numpy_int64(gt_sem)

        valid = (gt_sem >= 0) & (gt_sem < num_classes)
        pred_sem = pred_sem[valid]
        gt_sem = gt_sem[valid]

        for g, p in zip(gt_sem, pred_sem):
            if 0 <= p < num_classes:
                conf[g, p] += 1

        for c in range(num_classes):
            gt_count[c] += int((gt_sem == c).sum())
            pred_count[c] += int((pred_sem == c).sum())

    print('=' * 100)
    print('Confusion Matrix (rows = GT, cols = Pred)')
    print(conf)
    print('=' * 100)

    # 打印更直观表格
    df_conf = pd.DataFrame(
        conf,
        index=[f'GT_{CLASS_NAMES.get(i, i)}' for i in range(num_classes)],
        columns=[f'Pred_{CLASS_NAMES.get(i, i)}' for i in range(num_classes)]
    )
    print(df_conf)
    print('=' * 100)

    # 每类 recall / precision / IoU
    rows = []
    for c in range(num_classes):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        rows.append({
            'class_id': c,
            'class_name': CLASS_NAMES.get(c, str(c)),
            'gt_points': int(gt_count[c]),
            'pred_points': int(pred_count[c]),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'recall': recall,
            'precision': precision,
            'iou': iou,
        })

    df_metrics = pd.DataFrame(rows)
    print(df_metrics)
    print('=' * 100)

    # 重点看 J1 被分到哪里
    if 0 < num_classes:
        j1_row = conf[0]
        j1_total = j1_row.sum()
        if j1_total > 0:
            print('J1 GT distribution over predictions:')
            for c in range(num_classes):
                ratio = j1_row[c] / j1_total
                print(f'  GT J1 -> Pred {CLASS_NAMES.get(c, c)}: {j1_row[c]} ({ratio:.6f})')
        else:
            print('GT J1 点数为 0')

    # 输出
    if args.out:
        out_csv = args.out
    else:
        out_csv = osp.join(
            osp.dirname(args.checkpoint),
            'val_confusion_debug.csv'
        )

    out_dir = osp.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_conf.to_csv(out_csv, encoding='utf-8-sig')
    metrics_csv = out_csv.replace('.csv', '_metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False, encoding='utf-8-sig')

    print(f'Saved confusion matrix to: {out_csv}')
    print(f'Saved class metrics to   : {metrics_csv}')


if __name__ == '__main__':
    main()
