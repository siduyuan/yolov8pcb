# -*- coding: utf-8 -*-
"""YOLOv8s PCB 缺陷检测：分次增量训练（目标 100 轮，每次执行 20 轮，共 5 次）。

用法：在项目根目录执行 python train_incremental.py
- 第 1 次：从 yolov8s.pt 从头训练，总计划 100 轮，本次只跑 20 轮
- 第 2~5 次：自动从完整检查点续训，每次 20 轮
- 已跑满 100 轮时再执行会直接提示已完成

中断安全：每轮结束都会保存 last.pt，Ctrl+C / 关机后重新执行即可接着跑。

为什么需要 resume_full.pt：
ultralytics 在训练正常退出（含 stop 提前中断）后会调用 final_eval()，其中的
strip_optimizer() 会把 last.pt/best.pt 的 epoch 字段改成 -1 并清空 optimizer 状态，
导致下次 resume 误判为「已训练完成」。因此本脚本在 save_model 之后、strip 之前，
用 on_model_save 回调把完整的（含 optimizer + 正确 epoch）检查点备份为 resume_full.pt，
续训时优先使用它。
"""
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"d:\CodeForge\Project\TraePro\PCB_Yolov8")
DATA = ROOT / "dataset" / "data.yaml"
BASE_WEIGHTS = ROOT / "yolov8s.pt"
PROJECT = ROOT / "runs"
NAME = "pcb_v8s_100"

TOTAL_EPOCHS = 100    # 总目标轮数
SESSION_EPOCHS = 20   # 每次执行训练的轮数（5 次跑满 100 轮）

RUN_DIR = PROJECT / NAME
LAST = RUN_DIR / "weights" / "last.pt"
FULL = RUN_DIR / "weights" / "resume_full.pt"  # 完整检查点（含 optimizer，专用于续训）
BEST = RUN_DIR / "weights" / "best.pt"         # 兜底：last.pt 损坏时，退回最新的完整 best 检查点
CSV = RUN_DIR / "results.csv"


def finished_epochs():
    """从 results.csv 读取已完成轮数（续训时作为进度判断依据）。"""
    if not CSV.exists():
        return 0
    lines = [ln for ln in CSV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return max(len(lines) - 1, 0)  # 去掉表头


def ckpt_epoch(path: Path) -> int:
    """读取检查点里保存的 epoch 字段；失败或被 strip 时返回 -1。"""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", -1))
    except Exception:
        return -1


def repair_stripped_last(done: int):
    """旧脚本跑出的 last.pt 已被 strip（epoch=-1、无 optimizer）。权重仍在，修正 epoch 字段即可续训。

    只改 epoch 字段，权重不动；AdamW 动量会因 optimizer 已丢失而重置，影响很小（约 1 轮内的微小扰动）。
    """
    if done <= 0 or not LAST.exists():
        return
    if ckpt_epoch(LAST) >= 0:
        return  # 已是完整检查点，无需修复
    try:
        ck = torch.load(LAST, map_location="cpu", weights_only=False)
        ck["epoch"] = done - 1
        torch.save(ck, LAST)
        print(f"[train] 已修复 last.pt 的 epoch 字段 -> {done - 1}（{done} 轮已训练，权重保留）", flush=True)
    except Exception as e:
        print(f"[train] 修复 last.pt 失败: {e}", flush=True)


def pick_resume_source() -> Path | None:
    """在完整检查点、last.pt、best.pt 之间选一个 epoch 更大、可续训的来源。

    正常跑满一次会话后：last.pt 被 strip（epoch=-1），resume_full.pt 是完整的 → 选 resume_full.pt
    异常中断（Ctrl+C/崩溃）后：last.pt 保持完整且轮数更新 → 选 last.pt
    last.pt 损坏时：退回完整且更新的 best.pt（能少丢几轮）
    """
    candidates = [p for p in (FULL, LAST, BEST) if p.exists()]
    if not candidates:
        return None
    best = max(candidates, key=ckpt_epoch)
    return best if ckpt_epoch(best) >= 0 else None


def main():
    n_cpu = os.cpu_count() or 4
    n_threads = max(2, n_cpu - 2)  # 预留 2 核给系统
    torch.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    done = finished_epochs()
    print(f"[train] 已完成 {done}/{TOTAL_EPOCHS} 轮", flush=True)
    if done >= TOTAL_EPOCHS:
        print(f"[train] 已达目标轮数，无需继续训练。最优权重: {RUN_DIR / 'weights' / 'best.pt'}", flush=True)
        return

    target = min(done + SESSION_EPOCHS, TOTAL_EPOCHS)
    print(f"[train] 本次训练 {target - done} 轮（{done} -> {target}）", flush=True)

    state = {"stopped": False, "backed_up": False}

    def stop_after_session(trainer):
        """本次会话跑满 SESSION_EPOCHS 轮后在轮末停止（会正常做验证并保存权重）。"""
        if state["stopped"]:
            return
        if trainer.epoch + 1 - trainer.start_epoch >= SESSION_EPOCHS:
            state["stopped"] = True
            trainer.stop = True

    def backup_full(trainer):
        """save_model 落盘后、final_eval strip 之前，把完整检查点备份为 resume_full.pt。"""
        if trainer.stop and not state["backed_up"]:
            state["backed_up"] = True
            shutil.copyfile(trainer.last, FULL)
            print(f"[train] 已备份完整检查点（epoch={trainer.epoch}）-> {FULL}", flush=True)

    repair_stripped_last(done)  # 兼容旧脚本：修复被 strip 的 last.pt，避免浪费已训练的轮次

    resume_from = pick_resume_source()

    if resume_from is not None:
        ep = ckpt_epoch(resume_from)
        print(f"[train] 断点续训: {resume_from} (epoch={ep})", flush=True)
        model = YOLO(str(resume_from))
        model.add_callback("on_train_epoch_end", stop_after_session)
        model.add_callback("on_model_save", backup_full)
        # resume 时 epochs / 数据 / 增强参数沿用检查点，只覆盖设备、缓存、批处理等运行时参数
        model.train(resume=str(resume_from), device="cpu", batch=8, workers=0, cache="ram", plots=True)
    else:
        print(f"[train] 从头训练: {BASE_WEIGHTS}", flush=True)
        model = YOLO(str(BASE_WEIGHTS))
        model.add_callback("on_train_epoch_end", stop_after_session)
        model.add_callback("on_model_save", backup_full)
        model.train(
            data=str(DATA),
            epochs=TOTAL_EPOCHS,
            imgsz=640,
            batch=8,
            device="cpu",
            project=str(PROJECT),
            name=NAME,
            exist_ok=True,
            workers=0,
            cache="ram",
            plots=True,
            # ---------- 小目标增强 ----------
            mosaic=1.0,       # 四图拼接，增加小目标的密度与尺度多样性
            close_mosaic=10,  # 最后 10 轮关闭 mosaic，让收敛更稳定
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

    now = finished_epochs()
    print(f"\n[train] 本次会话结束，累计 {now}/{TOTAL_EPOCHS} 轮", flush=True)
    if now < TOTAL_EPOCHS:
        print(f"[train] 请再次执行 python train_incremental.py（还剩 {TOTAL_EPOCHS - now} 轮）", flush=True)
    else:
        print(f"[train] 100 轮已全部完成，最优权重: {RUN_DIR / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()