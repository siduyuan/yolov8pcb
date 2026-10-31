# -*- coding: utf-8 -*-
"""DeepPCB 数据预处理：bbox 转 YOLO 格式并划分 train/val/test = 7:2:1。"""
import random
import shutil
from pathlib import Path

SRC = Path(r"d:\CodeForge\Project\TraePro\PCB_Yolov8\DeepPCB\PCBData")
DST = Path(r"d:\CodeForge\Project\TraePro\PCB_Yolov8\dataset")

IMG_W = IMG_H = 640  # DeepPCB 全部为 640x640

# DeepPCB 的 type ID(1..6) -> YOLO class id(0..5)
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def collect_samples():
    samples = []  # (test_img_path, annot_txt_path)
    for not_dir in sorted(SRC.rglob("*_not")):
        if not not_dir.is_dir():
            continue
        img_dir = not_dir.parent / not_dir.name[:-4]  # 去掉后缀 "_not"
        for txt in sorted(not_dir.glob("*.txt")):
            test_img = img_dir / (txt.stem + "_test.jpg")
            if test_img.exists():
                samples.append((test_img, txt))
    return samples


def to_yolo(line):
    parts = line.strip().replace(",", " ").split()
    x1, y1, x2, y2, t = [float(v) for v in parts]
    cls = int(t) - 1
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return cls, cx, cy, w, h


def main():
    if DST.exists():
        shutil.rmtree(DST)
    samples = collect_samples()
    print(f"总样本数: {len(samples)}")

    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    stats = {}
    for split, items in splits.items():
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for test_img, txt in items:
            shutil.copy2(test_img, img_dir / test_img.name)
            lines = []
            for raw in txt.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                cls, cx, cy, w, h = to_yolo(raw)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[CLASS_NAMES[cls]] = stats.get(CLASS_NAMES[cls], 0) + 1
            (lbl_dir / (test_img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(items)} 张")

    print("\n各类缺陷实例数:")
    for k in CLASS_NAMES:
        print(f"  {k}: {stats.get(k, 0)}")

    yaml_path = DST / "data.yaml"
    yaml_path.write_text(
        "path: " + str(DST) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    print(f"\n已生成数据集配置: {yaml_path}")


if __name__ == "__main__":
    main()