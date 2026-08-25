"""Create a deterministic image/label manifest and fingerprint for a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze a YOLO dataset with SHA-256 manifests")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_label(path):
    classes, boxes = Counter(), 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"无效YOLO标签：{path}:{line_number}")
        cls = int(float(values[0]))
        if cls not in (0, 1):
            raise ValueError(f"类别越界：{path}:{line_number}: {cls}")
        classes[cls] += 1
        boxes += 1
    return boxes, classes


def main():
    args = parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"数据集不存在：{dataset}")
    if output.exists():
        raise SystemExit(f"输出目录已存在，停止覆盖：{output}")
    rows, split_counts, class_counts = [], {}, Counter()
    for split in SPLITS:
        image_dir, label_dir = dataset/split/"images", dataset/split/"labels"
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        labels = sorted(label_dir.rglob("*.txt"))
        expected_labels = set()
        split_classes, split_boxes = Counter(), 0
        for index, image in enumerate(images, 1):
            relative_image = image.relative_to(dataset)
            label = label_dir/image.relative_to(image_dir).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"缺少标签：{label}")
            expected_labels.add(label.resolve())
            boxes, classes = read_label(label)
            split_boxes += boxes
            split_classes.update(classes); class_counts.update(classes)
            image_hash, label_hash = sha256(image), sha256(label)
            rows.append({
                "数据版本": args.version, "数据划分": split,
                "图片相对路径": relative_image.as_posix(),
                "标签相对路径": label.relative_to(dataset).as_posix(),
                "图片字节数": image.stat().st_size, "标签字节数": label.stat().st_size,
                "图片SHA256": image_hash, "标签SHA256": label_hash,
                "标注框数量": boxes,
                "类别编号": ";".join(str(key) for key in sorted(classes)),
            })
            if index % 500 == 0:
                print(f"{split}: {index}/{len(images)}", flush=True)
        extras = [path for path in labels if path.resolve() not in expected_labels]
        if extras:
            raise ValueError(f"{split}存在{len(extras)}个无对应图片的标签")
        split_counts[split] = {"图片": len(images), "标签": len(labels), "标注框": split_boxes,
                               "类别": {str(k): v for k, v in sorted(split_classes.items())}}

    rows.sort(key=lambda row: row["图片相对路径"])
    aggregate = hashlib.sha256()
    for row in rows:
        canonical = "\t".join(str(row[key]) for key in (
            "图片相对路径", "标签相对路径", "图片字节数", "标签字节数", "图片SHA256", "标签SHA256"
        )) + "\n"
        aggregate.update(canonical.encode("utf-8"))
    output.mkdir(parents=True)
    fields = ["数据版本", "数据划分", "图片相对路径", "标签相对路径", "图片字节数", "标签字节数",
              "图片SHA256", "标签SHA256", "标注框数量", "类别编号"]
    version_slug = args.version.replace("/", "_").replace("\\", "_").replace(" ", "_")
    manifest = output/f"{version_slug}_file_manifest_sha256.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    manifest_hash = sha256(manifest)
    summary = {
        "数据版本": args.version,
        "数据集位置（本机）": str(dataset),
        "冻结状态": "frozen_for_baseline_training",
        "冻结时间UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "纳入范围": "仅train/val/test中的图片及对应.txt标签；不含缓存和说明文件",
        "文件对数量": len(rows), "图片数量": len(rows), "标签数量": len(rows),
        "标注框数量": sum(sum(value["类别"].values()) for value in split_counts.values()),
        "类别总数": {str(k): v for k, v in sorted(class_counts.items())},
        "数据划分": split_counts,
        "数据集总指纹SHA256": aggregate.hexdigest(),
        "清单文件SHA256": manifest_hash,
        "清单文件": manifest.name,
        "复核规则": f"重新运行本工具并比较数据集总指纹；如不同则必须创建新数据版本，禁止覆盖{args.version}",
    }
    (output/f"{version_slug}_freeze_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"README.md").write_text(
        f"# {args.version}数据冻结记录\n\n"
        f"该目录用于唯一标识基线训练所用的`{args.version}`。\n\n"
        f"- 文件对：{len(rows)}\n- 数据集总指纹：`{aggregate.hexdigest()}`\n"
        f"- 清单SHA-256：`{manifest_hash}`\n\n"
        f"冻结后不得直接修改{args.version}；发现问题时必须复制生成新版本，并重新分析、复检和冻结。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
