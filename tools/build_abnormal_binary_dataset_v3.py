"""从冻结的clean_v2构建“正常/异常”二分类clean_v3，不覆盖任何历史数据。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path


NORMAL_PREFIXES = {
    "nozzle_clean",
    "nozzle_extrusion_normal",
    "nozzle_no_extrusion",
}
ABNORMAL_PREFIXES = {
    "nozzle_slight_contamination",
    "nozzle_heavy_contamination",
    "nozzle_tip_wrapped",
}
DROP_PREFIXES = {"camera_occlusion"}
KNOWN_PREFIXES = NORMAL_PREFIXES | ABNORMAL_PREFIXES | DROP_PREFIXES
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def arguments():
    parser = argparse.ArgumentParser(description="构建正常/异常二分类clean_v3")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def prefix(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def read_rows(path: Path) -> list[list[str]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"无效YOLO标签：{path}:{number}")
        int(float(values[0]))
        coordinates = [float(value) for value in values[1:]]
        if any(not 0 <= value <= 1 for value in coordinates):
            raise ValueError(f"框坐标越界：{path}:{number}")
        rows.append(values)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> tuple[dict, Counter]:
    result, total = {}, Counter()
    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
        labels = sorted(label_dir.glob("*.txt"))
        classes, prefixes, boxes = Counter(), Counter(), 0
        for image in images:
            prefixes[prefix(image.stem)] += 1
        for label in labels:
            for row in read_rows(label):
                classes[int(float(row[0]))] += 1
                boxes += 1
        result[split] = {
            "图片": len(images), "标签": len(labels), "标注框": boxes,
            "类别": {str(k): v for k, v in sorted(classes.items())},
            "文件名前缀": dict(sorted(prefixes.items())),
        }
        total.update(prefixes)
    return result, total


def main():
    args = arguments()
    source, output, report = args.source.resolve(), args.output.resolve(), args.report_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"源数据集不存在：{source}")
    if output.exists() or report.exists():
        raise SystemExit(f"输出已存在，禁止覆盖：{output if output.exists() else report}")

    before, source_prefixes = collect(source)
    unknown = sorted(set(source_prefixes) - KNOWN_PREFIXES)
    missing = sorted(KNOWN_PREFIXES - set(source_prefixes))
    if unknown:
        raise ValueError(f"发现未登记类别前缀，停止自动处理：{unknown}")
    if missing:
        raise ValueError(f"映射表中的类别未在数据集中出现：{missing}")
    if source_prefixes["camera_occlusion"] != 1:
        raise ValueError(f"预期camera_occlusion恰好1张，实际{source_prefixes['camera_occlusion']}")

    report.mkdir(parents=True)
    shutil.copytree(source, output, copy_function=shutil.copy2)
    mapping_rows, deleted_rows = [], []

    for split in SPLITS:
        image_dir, label_dir = output / split / "images", output / split / "labels"
        for image in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            current_prefix = prefix(image.stem)
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"图片缺少标签：{image}")
            old_rows = read_rows(label)
            if current_prefix in DROP_PREFIXES:
                deleted_rows.append({
                    "数据划分": split,
                    "图片": image.relative_to(output).as_posix(),
                    "标签": label.relative_to(output).as_posix(),
                    "原始类别": current_prefix,
                    "处理": "删除图片及标签",
                    "原因": "业务确认camera_occlusion不纳入本项目",
                })
                image.unlink()
                label.unlink()
                continue

            new_class = 0 if current_prefix in NORMAL_PREFIXES else 1
            new_rows = [[str(new_class), *row[1:]] for row in old_rows]
            if [row[1:] for row in old_rows] != [row[1:] for row in new_rows]:
                raise RuntimeError(f"框坐标意外变化：{label}")
            label.write_text("\n".join(" ".join(row) for row in new_rows) + "\n", encoding="utf-8")
            mapping_rows.append({
                "数据划分": split,
                "图片": image.relative_to(output).as_posix(),
                "标签": label.relative_to(output).as_posix(),
                "原始类别": current_prefix,
                "旧类别编号": ";".join(row[0] for row in old_rows),
                "新类别编号": str(new_class),
                "新类别名称": "nozzle_normal" if new_class == 0 else "nozzle_abnormal",
                "标注框数量": len(new_rows),
                "框坐标是否保持不变": "是",
            })

    after, after_prefixes = collect(output)
    if "camera_occlusion" in after_prefixes:
        raise RuntimeError("clean_v3中仍存在camera_occlusion")
    for split, values in after.items():
        if values["图片"] != values["标签"] or values["图片"] != values["标注框"]:
            raise RuntimeError(f"{split}图片、标签或单框数量不一致：{values}")
        if set(map(int, values["类别"])) - {0, 1}:
            raise RuntimeError(f"{split}存在类别越界")

    write_csv(
        report / "标签重映射审计清单.csv", mapping_rows,
        ["数据划分", "图片", "标签", "原始类别", "旧类别编号", "新类别编号", "新类别名称", "标注框数量", "框坐标是否保持不变"],
    )
    write_csv(
        report / "camera_occlusion删除清单.csv", deleted_rows,
        ["数据划分", "图片", "标签", "原始类别", "处理", "原因"],
    )
    summary = {
        "数据版本": "dataset_2_abnormal_binary_clean_v3",
        "源数据集": str(source),
        "输出数据集": str(output),
        "业务定义": {"0": "nozzle_normal（无异常）", "1": "nozzle_abnormal（异常）"},
        "映射": {
            "0": sorted(NORMAL_PREFIXES),
            "1": sorted(ABNORMAL_PREFIXES),
            "删除": sorted(DROP_PREFIXES),
        },
        "处理前": before,
        "处理后": after,
        "重映射图片及标签": len(mapping_rows),
        "删除camera_occlusion": len(deleted_rows),
        "框坐标变化": 0,
        "源clean_v2是否修改": False,
    }
    (report / "clean_v3构建汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "CLEANING_PROVENANCE.txt").write_text(
        "数据集版本：dataset_2_abnormal_binary_clean_v3\n"
        f"源数据集：{source}\n"
        "业务标签：0=nozzle_normal；1=nozzle_abnormal。\n"
        "操作：继承clean_v2的去重、抽帧和数据划分；删除唯一camera_occlusion；按文件名前缀重映射类别；框坐标不变。\n"
        f"构建报告：{report}\n源clean_v2未修改。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
