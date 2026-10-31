# -*- coding: utf-8 -*-
"""一次性跑完剩余训练轮次（从当前检查点续训到 100 轮）。

与 train_incremental.py 的区别：本脚本不设「每次 20 轮」的 stop，
而是直接续训到 TOTAL_EPOCHS，适合挂机一次跑完。

用法：python train_rest.py

崩溃安全：
- 每轮结束自动保存 last.pt（ultralytics 自动）
- last.pt 因崩溃损坏时，pick_resume_source 会退回完整的 best.pt / resume_full.pt
- 中途崩溃/关机后，重新执行本脚本即可接着跑，无需额外操作
"""
import os

import torch
from ultralytics import YOLO

from train_incremental import (
    TOTAL_EPOCHS,
    RUN_DIR,
    ckpt_epoch,
    pick_resume_source,
    repair_stripped_last,
)

MIN_PER_EPOCH = 10.7  # 据历史统计，单轮均值约 10.7 分钟


def main():
    n_cpu = os.cpu_count() or 4
    n_threads = max(2, n_cpu - 2)
    torch.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    # 兼容旧脚本：若 last.pt 被 strip（epoch=-1）则修正其 epoch 字段
    repair_stripped_last(len(open(RUN_DIR / "results.csv", encoding="utf-8").read().splitlines()) - 1)

    resume_from = pick_resume_source()
    if resume_from is None:
        print("[train_rest] 未找到可续训检查点，请先确认 runs/pcb_v8s_100/weights/ 下存在权重", flush=True)
        return

    ep = ckpt_epoch(resume_from)
    remaining = TOTAL_EPOCHS - (ep + 1)  # resume 后从 start_epoch=ep+1 开始训练
    if remaining <= 0:
        print(f"[train_rest] 已训练完成，无需继续。best 权重: {RUN_DIR / 'weights' / 'best.pt'}", flush=True)
        return

    hours = remaining * MIN_PER_EPOCH / 60
    print(f"[train_rest] 续训来源: {resume_from} (epoch={ep})", flush=True)
    print(f"[train_rest] 剩余 {remaining} 轮，预计 {hours:.1f} 小时（按 {MIN_PER_EPOCH} 分钟/轮）", flush=True)

    model = YOLO(str(resume_from))
    model.train(resume=str(resume_from), device="cpu", batch=8, workers=0, cache="ram", plots=True)

    print(f"\n[train_rest] 训练结束，累计完成 {ckpt_epoch(RUN_DIR / 'weights' / 'best.pt') + 1}/{TOTAL_EPOCHS} 轮（以 best.pt 为准）", flush=True)
    print(f"[train_rest] best 权重: {RUN_DIR / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()