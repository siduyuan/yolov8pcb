# -*- coding: utf-8 -*-
"""关闭 mosaic 的短轮数重训（从 yolov8s.pt 从头训练 40 轮）。

背景：100 轮实验（mosaic=1.0 全开）best mAP50-95=0.690，与 10 轮基线 0.679 几乎持平，
说明瓶颈不在训练轮数。PCB 缺陷是规则几何图形，mosaic 四图拼接会破坏其规律性，
故本次实验关闭 mosaic（mosaic=0.0），仅保留单图增强，并缩短到 40 轮观察是否更优。

与 pcb_v8s_100 的差异：
  - mosaic: 1.0 -> 0.0（关闭，唯一关键变量）
  - epochs: 100 -> 40
  - 其余（scale/fliplr/hsv_v、lr、优化器、batch、imgsz）保持一致

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
        # ---------- 关闭 mosaic，仅保留单图增强 ----------
        mosaic=0.0,       # 关键变量：关闭四图拼接，避免破坏 PCB 缺陷的规则几何特征
        scale=0.5,        # 随机缩放，制造更多小尺度目标
        translate=0.1,
        fliplr=0.5,
        degrees=0.0,      # PCB 版面方向固定，不做旋转
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,        # 数据集为灰度图，色调/饱和度无意义
        hsv_s=0.0,
        hsv_v=0.3,        # 只保留亮度扰动
    )

    print(f"\n[train_short] 训练结束，best 权重: {PROJECT / NAME / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()