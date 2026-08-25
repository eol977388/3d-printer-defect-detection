"""Verify a frozen YOLO dataset against its SHA-256 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


FIELDS = ("图片相对路径", "标签相对路径", "图片字节数", "标签字节数", "图片SHA256", "标签SHA256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_line(row: dict) -> str:
    return "\t".join(str(row[key]) for key in FIELDS) + "\n"


def verify_dataset(dataset: Path, freeze_dir: Path, progress: bool = True) -> dict:
    dataset, freeze_dir = dataset.resolve(), freeze_dir.resolve()
    manifests = sorted(freeze_dir.glob("*_file_manifest_sha256.csv"))
    summaries = sorted(freeze_dir.glob("*_freeze_summary.json"))
    if len(manifests) != 1 or len(summaries) != 1:
        raise FileNotFoundError(f"冻结目录必须各有1个清单和汇总，实际为{len(manifests)}和{len(summaries)}：{freeze_dir}")
    manifest_path, summary_path = manifests[0], summaries[0]
    if not dataset.is_dir():
        raise FileNotFoundError(f"数据集不存在：{dataset}")
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"冻结清单不完整：{freeze_dir}")
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing, changed, current_rows = [], [], []
    expected_files = set()
    for index, expected in enumerate(rows, 1):
        image_rel, label_rel = expected["图片相对路径"], expected["标签相对路径"]
        image, label = dataset / image_rel, dataset / label_rel
        expected_files.update((Path(image_rel).as_posix(), Path(label_rel).as_posix()))
        absent = [rel for rel, path in ((image_rel, image), (label_rel, label)) if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        actual = dict(expected)
        actual.update({"图片字节数": str(image.stat().st_size), "标签字节数": str(label.stat().st_size),
                       "图片SHA256": sha256(image), "标签SHA256": sha256(label)})
        differences = [key for key in FIELDS[2:] if str(expected[key]) != str(actual[key])]
        if differences:
            changed.append({"图片": image_rel, "标签": label_rel, "变化字段": differences})
        current_rows.append(actual)
        if progress and index % 500 == 0:
            print(f"已验证 {index}/{len(rows)}", flush=True)
    actual_files = set()
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            folder = dataset / split / kind
            if folder.is_dir():
                for path in folder.rglob("*"):
                    if path.is_file() and (kind == "images" or path.suffix.lower() == ".txt"):
                        # Ignore YOLO cache files and count only files represented by the frozen scope.
                        if path.suffix.lower() != ".cache":
                            actual_files.add(path.relative_to(dataset).as_posix())
    added = sorted(actual_files - expected_files)
    aggregate = hashlib.sha256()
    for row in sorted(current_rows, key=lambda value: value["图片相对路径"]):
        aggregate.update(canonical_line(row).encode("utf-8"))
    actual_fingerprint = aggregate.hexdigest() if len(current_rows) == len(rows) else None
    expected_fingerprint = summary["数据集总指纹SHA256"]
    passed = not missing and not added and not changed and actual_fingerprint == expected_fingerprint
    return {"通过": passed, "冻结版本": summary["数据版本"], "预期文件对": len(rows),
            "成功验证文件对": len(current_rows), "缺失文件": sorted(set(missing)), "新增文件": added,
            "内容或大小变化": changed, "预期总指纹": expected_fingerprint,
            "当前总指纹": actual_fingerprint}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练前验证冻结数据集")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="可选：保存本次验证JSON")
    args = parser.parse_args()
    try:
        result = verify_dataset(args.dataset, args.freeze)
    except Exception as error:
        print(f"FAIL：冻结验证无法完成：{error}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["通过"]:
        print(f"PASS：数据集与{result['冻结版本']}冻结记录完全一致")
        print(f"数据集总指纹：{result['当前总指纹']}")
        return 0
    print("FAIL：数据集已发生变化，禁止训练", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
