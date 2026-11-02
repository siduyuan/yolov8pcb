# -*- coding: utf-8 -*-
"""测试集评估脚本（交付版，相对路径，任意目录可直接运行）。

用法：
  python test.py             # 普通评估 + TTA + 置信度扫描（CPU 约 6-8 分钟）
  python test.py --no-tta    # 跳过 TTA，只做普通评估 + 置信度扫描
  python test.py --weights 其他权重.pt   # 指定其他权重

依赖：pip install ultralytics torch
输出：控制台打印结果，summary.txt 保存在本目录 runs/eval_final/
"""
import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

# 基于脚本所在目录，保证整个交付文件夹搬到任意位置都能跑
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data.yaml"
DEFAULT_WEIGHTS = SCRIPT_DIR / "best.pt"
PROJECT = SCRIPT_DIR / "runs"
EVAL_DIR = PROJECT / "eval_final"

CONF_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def run_val(model, name, augment=False, conf=0.001, plots=False):
    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(PROJECT),
        name=name,
        exist_ok=True,
        workers=0,
        augment=augment,
        conf=conf,
        plots=plots,
        verbose=False,
    )
    return metrics.box


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    n_cpu = os.cpu_count() or 4
    torch.set_num_threads(max(2, n_cpu - 2))

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"权重不存在: {weights}")

    print(f"[test] 权重: {weights}", flush=True)
    model = YOLO(str(weights))
    names = model.names
    lines = [f"权重: {weights}", ""]

    # ---------- 1. 普通评估 ----------
    base = run_val(model, "test_base", augment=False, plots=True)
    print("\n=== 1. 普通评估（测试集）===")
    print(f"Precision: {base.mp:.4f}  Recall: {base.mr:.4f}  "
          f"mAP50: {base.map50:.4f}  mAP50-95: {base.map:.4f}  F1: {f1_score(base.mp, base.mr):.4f}")
    lines.append(f"[普通评估] P={base.mp:.4f} R={base.mr:.4f} mAP50={base.map50:.4f} "
                 f"mAP50-95={base.map:.4f} F1={f1_score(base.mp, base.mr):.4f}")

    print(f"\n{'class':<12}{'AP50':>8}{'AP50-95':>10}")
    idxs = list(getattr(base, "ap_class_index", range(len(names))))
    for i, ci in enumerate(idxs):
        print(f"{names[int(ci)]:<12}{base.ap50[i]:>8.4f}{base.ap[i]:>10.4f}")
        lines.append(f"  {names[int(ci)]}: AP50={base.ap50[i]:.4f} AP50-95={base.ap[i]:.4f}")

    # ---------- 2. TTA 评估 ----------
    if not args.no_tta:
        tta = run_val(model, "test_tta", augment=True, plots=False)
        print("\n=== 2. TTA 测试增强评估 ===")
        print(f"Precision: {tta.mp:.4f}  Recall: {tta.mr:.4f}  "
              f"mAP50: {tta.map50:.4f}  mAP50-95: {tta.map:.4f}  F1: {f1_score(tta.mp, tta.mr):.4f}")
        print(f"相比普通评估: mAP50 {tta.map50 - base.map50:+.4f}  mAP50-95 {tta.map - base.map:+.4f}")
        lines.append(f"[TTA 评估] P={tta.mp:.4f} R={tta.mr:.4f} mAP50={tta.map50:.4f} "
                     f"mAP50-95={tta.map:.4f} F1={f1_score(tta.mp, tta.mr):.4f}")

    # ---------- 3. 置信度阈值调优 ----------
    print("\n=== 3. 置信度阈值扫描 ===")
    print(f"{'conf':<8}{'P':>8}{'R':>8}{'F1':>8}")
    lines.append("")
    lines.append("[置信度扫描] conf / P / R / F1")
    best = None
    for conf in CONF_LIST:
        box = run_val(model, f"test_conf_{int(conf * 100):03d}", conf=conf, plots=False)
        f1 = f1_score(box.mp, box.mr)
        print(f"{conf:<8.2f}{box.mp:>8.4f}{box.mr:>8.4f}{f1:>8.4f}")
        lines.append(f"  {conf:.2f} / {box.mp:.4f} / {box.mr:.4f} / {f1:.4f}")
        if best is None or f1 > best[1]:
            best = (conf, f1, box.mp, box.mr)

    print(f"\n最佳 F1 阈值: conf={best[0]:.2f}  F1={best[1]:.4f}  P={best[2]:.4f}  R={best[3]:.4f}")
    lines.append("")
    lines.append(f"[最佳阈值] conf={best[0]:.2f} F1={best[1]:.4f} P={best[2]:.4f} R={best[3]:.4f}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[test] 汇总已保存: {EVAL_DIR / 'summary.txt'}", flush=True)


if __name__ == "__main__":
    main()