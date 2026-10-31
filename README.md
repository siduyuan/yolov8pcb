# 基于 YOLOv8 的 PCB 缺陷检测

> 课程设计（软件阶段）项目：使用 YOLOv8 对 PCB 电路板进行 6 类缺陷的目标检测。

**作者：孙德勇　学号：2023203754**

## 项目简介

本项目的任务类型为**目标检测（Object Detection）**，使用 YOLOv8 算法对 PCB 图像进行缺陷检测，输出缺陷类别及其坐标框 `(x, y, w, h)`。输入一张 PCB 图像，模型在图上画出缺陷框、标注缺陷名称与置信度。

本阶段聚焦数据集准备、模型训练、验证评估与推理可视化，不涉及工业相机采集与硬件部署。

## 缺陷类别（6 类）

基于 DeepPCB / PCB-Defect-Dataset 数据集的真实 6 类：

| 类别 | 中文 | 英文 |
|------|------|------|
| 0 | 开路 | open circuit |
| 1 | 短路 | short |
| 2 | 鼠咬 | mouse bite |
| 3 | 毛刺 | spur |
| 4 | 多余铜箔（杂铜） | spurious copper |
| 5 | 缺孔（针孔） | missing hole / pin hole |

## 目录结构

```
PCB_Yolov8/
├── prepare_dataset.py        # DeepPCB 数据预处理（bbox 转 YOLO 格式，划分 train/val/test = 7:2:1）
├── train.py                  # YOLOv8 CPU 完整训练入口
├── train_incremental.py      # 增量训练脚本
├── train_rest.py             # 断点续训脚本
├── test.py                   # 推理测试脚本
├── 课程设计-软件阶段总结.md   # 软件阶段总结文档
├── 课程设计.pptx             # 课程设计汇报 PPT
├── DeepPCB/                  # 原始数据集（已入库）
├── dataset/                  # 处理后数据集（已入库，含 data.yaml）
└── runs/                     # 训练输出（不入库）
```

## 流程主线

数据集准备 → 数据标注 → 数据集划分 → 环境搭建 → 模型训练 → 模型验证评估 → 推理测试（检测 + 定位 + 分类）→ 结果可视化

## 环境依赖

- Python 3.x
- PyTorch
- ultralytics（YOLOv8）

## 使用说明

1. 准备数据集：`DeepPCB/` 原始数据与 `dataset/` 处理后数据（含 `data.yaml`）均已入库，克隆后可直接使用；如需重新划分可运行 `prepare_dataset.py`。
2. 训练模型：运行 `train.py`（或根据需求使用 `train_incremental.py` / `train_rest.py`）。
3. 推理测试：运行 `test.py`。

## 许可证

本项目基于 [GPL v3.0](LICENSE) 许可证发布。