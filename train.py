# -*- coding: utf-8 -*-
"""YOLOv8 CPU 完整训练（10 epoch，供长期挂机）。

保活说明：由启动脚本作为独立进程运行（Start-Process），
日志重定向到 train_output.log / train_error.log，进程不依附于交互式终端。
"""
import os
import sys
import traceback

import torch
from ultralytics import YOLO

DATA = r"d:\CodeForge\Project\TraePro\PCB_Yolov8\dataset\data.yaml"
WEIGHTS = r"d:\CodeForge\Project\TraePro\PCB_Yolov8\yolov8s.pt"
PROJECT = r"d:\CodeForge\Project\TraePro\PCB_Yolov8\runs"


def main():
    n_cpu = os.cpu_count() or 4
    n_threads = max(2, n_cpu - 2)  # 预留 2 核给系统，防止卡死
    torch.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    print(f"[train] 逻辑CPU核={n_cpu}, torch线程数={n_threads}", flush=True)

    model = YOLO(WEIGHTS)
    model.train(
        data=DATA,
        epochs=10,
        imgsz=640,
        batch=8,
        device="cpu",
        project=PROJECT,
        name="pcb_yolov8s_full",
        workers=0,
        exist_ok=True,
    )
    print("[train] 训练完成", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)