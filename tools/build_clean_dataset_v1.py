"""Build the first traceable binary-label clean dataset without changing raw data.

Operations:
1. Copy dataset_2 to a new directory.
2. Apply the four manually reviewed binary-class corrections.
3. Resolve the 916 exact-duplicate pairs by keeping the tighter annotation source:
   nozzle_no_extrusion (914 pairs) or nozzle_slight_contamination (2 pairs).
4. Write Chinese audit tables and a machine-readable summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


KEEP_PREFIXES = {"nozzle_no_extrusion", "nozzle_slight_contamination"}
DROP_PREFIXES = {"nozzle_clean", "nozzle_extrusion_normal"}
MANUAL_CLASS_FIXES = {
    "nozzle_extrusion_normal_0004": 1,
    "nozzle_heavy_contamination_0001": 1,
    "nozzle_slight_contamination_0006": 1,
    "nozzle_tip_wrapped_0728": 0,
}


def args_parser():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="构建二分类清洗数据集v1")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--duplicates-csv",
        type=Path,
        default=project / "reports/data_analysis/raw_v1/tables/exact_duplicate_images.csv",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project / "reports/data_cleaning/clean_v1",
    )
    return parser.parse_args()


def prefix(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def label_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def area(label: Path) -> float:
    rows = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError(f"本规则要求每张图恰好一个YOLO框：{label}")
    return float(rows[0][3]) * float(rows[0][4])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_source(path: Path, source: Path) -> Path:
    try:
        return path.resolve().relative_to(source.resolve())
    except ValueError as exc:
        raise ValueError(f"清单中的文件不属于源数据集：{path}") from exc


def load_groups(csv_path: Path):
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    groups = defaultdict(list)
    for row in rows[1:]:
        groups[row[0]].append({"group": int(row[0]), "hash": row[1], "split": row[2], "image": Path(row[3])})
    if len(groups) != 916 or any(len(items) != 2 for items in groups.values()):
        raise ValueError(f"预期916个两文件重复组，实际{len(groups)}组")
    return groups


def choose_pair(items, source):
    enriched = []
    for item in items:
        image = item["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        label = label_for(image)
        if not label.is_file():
            raise FileNotFoundError(label)
        if sha256(image) != item["hash"]:
            raise ValueError(f"图片哈希已变化，停止处理：{image}")
        enriched.append({**item, "label": label, "prefix": prefix(image.stem), "area": area(label)})

    keep_candidates = [row for row in enriched if row["prefix"] in KEEP_PREFIXES]
    drop_candidates = [row for row in enriched if row["prefix"] in DROP_PREFIXES]
    if len(keep_candidates) != 1 or len(drop_candidates) != 1:
        names = [row["prefix"] for row in enriched]
        raise ValueError(f"重复组{items[0]['group']}不符合已审核的前缀规则：{names}")
    keep, drop = keep_candidates[0], drop_candidates[0]
    if keep["area"] >= drop["area"]:
        raise ValueError(f"重复组{items[0]['group']}的拟保留框并非小框")
    keep["relative"] = relative_to_source(keep["image"], source)
    drop["relative"] = relative_to_source(drop["image"], source)
    return keep, drop


def rewrite_class(path: Path, new_class: int):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    output, old_classes = [], []
    for line in lines:
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"无效YOLO标签：{path}: {line}")
        old_classes.append(int(float(values[0])))
        output.append(" ".join([str(new_class), *values[1:]]))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return old_classes


def count_pairs(root: Path):
    result = {}
    for split in ("train", "val", "test"):
        images = list((root / split / "images").glob("*"))
        labels = list((root / split / "labels").glob("*.txt"))
        result[split] = {"图片": len([p for p in images if p.is_file()]), "标签": len(labels)}
    return result


def main():
    args = args_parser()
    source, output, report = args.source.resolve(), args.output.resolve(), args.report_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"源数据集不存在：{source}")
    if output.exists():
        raise SystemExit(f"输出目录已存在，为防止覆盖已停止：{output}")
    if report.exists():
        raise SystemExit(f"报告目录已存在，为防止覆盖已停止：{report}")

    groups = load_groups(args.duplicates_csv)
    decisions = []
    for group_id in sorted(groups, key=int):
        keep, drop = choose_pair(groups[group_id], source)
        decisions.append((keep, drop))

    report.mkdir(parents=True)
    fields = ["重复组编号", "SHA-256", "保留划分", "保留图片", "保留标签", "保留来源前缀", "保留框面积",
              "删除划分", "删除图片", "删除标签", "删除来源前缀", "删除框面积", "是否跨划分", "处理理由"]
    decision_rows = []
    for keep, drop in decisions:
        decision_rows.append({
            "重复组编号": keep["group"], "SHA-256": keep["hash"], "保留划分": keep["split"],
            "保留图片": str(keep["relative"]), "保留标签": str(label_for(keep["relative"])),
            "保留来源前缀": keep["prefix"], "保留框面积": f"{keep['area']:.9f}",
            "删除划分": drop["split"], "删除图片": str(drop["relative"]),
            "删除标签": str(label_for(drop["relative"])), "删除来源前缀": drop["prefix"],
            "删除框面积": f"{drop['area']:.9f}", "是否跨划分": "是" if keep["split"] != drop["split"] else "否",
            "处理理由": "完全相同图片仅保留一份；人工确认小框完整且更规范",
        })
    with (report / "916组完全重复图片处理清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decision_rows)

    before = count_pairs(source)
    shutil.copytree(source, output, copy_function=shutil.copy2)

    removed = []
    for keep, drop in decisions:
        image_out = output / drop["relative"]
        label_out = output / label_for(drop["relative"])
        image_out.unlink()
        label_out.unlink()
        removed.extend([str(drop["relative"]), str(label_for(drop["relative"]))])

    fixes = []
    for stem, new_class in MANUAL_CLASS_FIXES.items():
        matches = list(output.glob(f"*/labels/{stem}.txt"))
        if len(matches) != 1:
            raise ValueError(f"人工复核标签应唯一存在：{stem}，实际{len(matches)}个")
        old = rewrite_class(matches[0], new_class)
        fixes.append({"文件": str(matches[0].relative_to(output)), "原类别": ";".join(map(str, old)),
                      "修正类别": new_class, "依据": "人工复核"})
    with (report / "4项人工复核标签修正.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["文件", "原类别", "修正类别", "依据"])
        writer.writeheader()
        writer.writerows(fixes)

    after = count_pairs(output)
    hashes = defaultdict(list)
    for image in output.glob("*/images/*"):
        if image.is_file():
            hashes[sha256(image)].append(image)
    remaining_exact = {key: value for key, value in hashes.items() if len(value) > 1}
    if remaining_exact:
        raise RuntimeError(f"清洗后仍有{len(remaining_exact)}组完全重复图片")
    for split, counts in after.items():
        if counts["图片"] != counts["标签"]:
            raise RuntimeError(f"{split}图片标签数量不一致：{counts}")

    combo = Counter(f"{keep['split']}→{drop['split']}" for keep, drop in decisions)
    kept_prefix = Counter(keep["prefix"] for keep, _ in decisions)
    summary = {
        "源数据集": str(source), "输出数据集": str(output), "创建方式": "复制后清洗，源数据集未修改",
        "处理的完全重复组": len(decisions), "删除图片": len(decisions), "删除标签": len(decisions),
        "人工修正标签": len(fixes), "处理前": before, "处理后": after,
        "保留来源前缀": dict(kept_prefix), "保留划分到删除划分": dict(sorted(combo.items())),
        "清洗后完全重复组": 0,
    }
    (report / "clean_v1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "CLEANING_PROVENANCE.txt").write_text(
        "数据集版本：clean_v1（二分类）\n"
        f"源数据集：{source}\n"
        "操作：应用4项人工复核标签修正；916组完全相同图片保留小框版本并删除另一份。\n"
        f"完整记录：{report}\n"
        "原始数据集未修改。\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
