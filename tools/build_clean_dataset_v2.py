"""Apply the reviewed v2 dry-run plan to a copy of clean_v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="构建二分类清洗数据集v2")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_for(image):
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def rewrite_class(path, cls):
    output = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"无效YOLO标签：{path}")
            output.append(" ".join([str(cls), *values[1:]]))
    path.write_text("\n".join(output)+"\n", encoding="utf-8")


def counts(root):
    result = {}
    for split in ("train", "val", "test"):
        images = [p for p in (root/split/"images").glob("*") if p.is_file()]
        labels = list((root/split/"labels").glob("*.txt"))
        classes = Counter()
        for label in labels:
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip(): classes[int(float(line.split()[0]))] += 1
        result[split] = {"图片": len(images), "标签": len(labels), "类别": dict(sorted(classes.items()))}
    return result


def main():
    args = parse_args()
    source, output, report = args.source.resolve(), args.output.resolve(), args.report_dir.resolve()
    if output.exists() or report.exists():
        raise SystemExit(f"输出已存在，停止覆盖：{output if output.exists() else report}")
    rows = list(csv.DictReader(args.plan.open(encoding="utf-8-sig")))
    if len(rows) != 1095:
        raise ValueError(f"预期1095条预案，实际{len(rows)}")
    before = counts(source)
    report.mkdir(parents=True)
    shutil.copytree(source, output, copy_function=shutil.copy2)

    deleted, moved = [], []
    # Delete first so target filenames are guaranteed available after preflight.
    for row in rows:
        rel = Path(row["图片"])
        if row["处理决定"] == "拟删除":
            image, label = output/rel, output/label_for(rel)
            image.unlink(); label.unlink()
            deleted.append({"图片": str(rel), "标签": str(label_for(rel)), "代表帧": row["对应代表帧"],
                            "理由": row["处理理由"]})
    for row in rows:
        if row["处理决定"] != "拟保留":
            continue
        rel, target = Path(row["图片"]), row["拟数据划分"]
        if target not in {"train", "val", "test"}:
            raise ValueError(f"无效目标划分：{target}")
        if rel.parts[0] == target:
            continue
        old_image, old_label = output/rel, output/label_for(rel)
        new_image = output/target/"images"/old_image.name
        new_label = output/target/"labels"/old_label.name
        if new_image.exists() or new_label.exists():
            raise FileExistsError(f"移动目标冲突：{new_image}")
        shutil.move(old_image, new_image); shutil.move(old_label, new_label)
        moved.append({"图片": str(rel), "原划分": rel.parts[0], "新划分": target,
                      "新图片": str(new_image.relative_to(output))})

    fixed = list(output.glob("*/labels/nozzle_heavy_contamination_0011.txt"))
    if len(fixed) != 1:
        raise ValueError(f"第5项标签应唯一存在，实际{len(fixed)}")
    rewrite_class(fixed[0], 0)
    after = counts(output)
    for split, data in after.items():
        if data["图片"] != data["标签"]:
            raise RuntimeError(f"{split}图片标签数量不一致")
    hashes = defaultdict(list)
    for image in output.glob("*/images/*"):
        if image.is_file(): hashes[sha256(image)].append(str(image.relative_to(output)))
    duplicate_hashes = {key: value for key, value in hashes.items() if len(value) > 1}
    if duplicate_hashes:
        raise RuntimeError(f"v2仍有{len(duplicate_hashes)}组完全重复图片")
    cluster_targets = defaultdict(set)
    for row in rows:
        if row["处理决定"] == "拟保留": cluster_targets[row["相似簇编号"]].add(row["拟数据划分"])
    if any(len(x) != 1 for x in cluster_targets.values()):
        raise RuntimeError("存在相似簇跨数据划分")

    for name, records, fields in (
        ("实际删除清单.csv", deleted, ["图片", "标签", "代表帧", "理由"]),
        ("实际移动清单.csv", moved, ["图片", "原划分", "新划分", "新图片"]),
    ):
        with (report/name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary = {"源数据集": str(source), "输出数据集": str(output), "处理前": before, "处理后": after,
               "删除高度冗余图片": len(deleted), "移动代表帧": len(moved), "第5项人工标签修正": 1,
               "完全重复组": 0, "247个原始跨划分相似簇是否均在单一划分": True,
               "说明": "从clean_v1复制构建，clean_v1未修改"}
    (report/"clean_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"CLEANING_PROVENANCE.txt").write_text(
        f"数据集版本：clean_v2（二分类）\n源数据集：{source}\n"
        "操作：第5项人工标签修正；删除112张高度冗余帧；247个跨划分相似簇整体重划分。\n"
        f"报告：{report}\n源数据集未修改。\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
