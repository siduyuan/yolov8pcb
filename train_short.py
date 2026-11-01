# -*- coding: utf-8 -*-
"""关闭 mosaic 的短轮数重训（从 yolov8s.pt 从头训练 40 轮）。

背景：100 轮实验（mosaic=1.0 全开 + lr0=0.001）best mAP50-95=0.690，与 10 轮基线 0.679
几乎持平，说明瓶颈不在训练轮数。实测数据定位到三个确定问题：
  1) lr0=0.001 对 AdamW 过高，参数震荡（mAP50-95 标准差高达 0.129）
  2) mosaic=1.0 破坏 PCB 规则几何缺陷的形态，定位精度低
  3) 100 轮在 1050 张数据上过拟合（train box loss 0.665 vs val box loss 均值 1.73）

与 pcb_v8s_100 的差异（三处确定性修正）：
  - mosaic:  1.0 -> 0.0（关闭四图拼接）
  - lr0:     0.001 -> 0.0005（AdamW 更稳，抗震荡）
  - epochs:  100 -> 40（够收敛，避免过拟合）
  - erasing: 0.4(默认) -> 0.0（关闭 Random Erasing，灰度 PCB 上是额外干扰）
  - 其余（scale/fliplr/hsv_v、优化器、batch、imgsz）保持一致

用法（全程由你手动执行，共约 40 × 10.5 ≈ 7 小时）：
  python train_short.py

产出：runs/pcb_v8s_nomosaic/（权重 best.pt / last.pt，曲线 results.png 等）
"""
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"d:\CodeForge\Project\TraePro\PCB_Yolov8")
DATA = ROOT / "dataset" / "data.yaml"
BASE_WEIGHTS = ROOT / "yolov8s.pt"
PROJECT = ROOT / "runs"
NAME = "pcb_v8s_nomosaic"  # 独立目录，不覆盖 100 轮实验

EPOCHS = 40


def main():
    n_cpu = os.cpu_count() or 4
    n_threads = max(2, n_cpu - 2)
    torch.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    print(f"[train_short] 从头训练 YOLOv8s，epochs={EPOCHS}，mosaic=0（关闭）", flush=True)

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        workers=0,
        cache="ram",
        plots=True,
        lr0=0.0005,       # 确定性修正：AdamW 下更稳的学习率（100 轮实验的 0.001 偏高致震荡）
        # ---------- 关闭 mosaic，仅保留单图增强 ----------
        mosaic=0.0,       # 确定性修正：关闭四图拼接，避免破坏 PCB 缺陷的规则几何特征
        scale=0.5,        # 随机缩放，制造更多小尺度目标
        translate=0.1,
        fliplr=0.5,
        degrees=0.0,      # PCB 版面方向固定，不做旋转
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,        # 数据集为灰度图，色调/饱和度无意义
        hsv_s=0.0,
        hsv_v=0.3,        # 只保留亮度扰动
        erasing=0.0,      # 确定性修正：关闭 Random Erasing（默认 0.4，灰度 PCB 上是额外干扰）
    )

    print(f"\n[train_short] 训练结束，best 权重: {PROJECT / NAME / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()