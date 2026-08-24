"""Find near-duplicate images using 64-bit pHash followed by Gaussian-window SSIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="pHash + SSIM视觉近重复检测")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phash-similarity", type=float, default=0.85)
    parser.add_argument("--ssim", type=float, default=0.85)
    parser.add_argument("--ssim-size", type=int, default=256)
    return parser.parse_args()


def dct_matrix(n):
    x = np.arange(n)
    k = x[:, None]
    matrix = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / n)


DCT32 = dct_matrix(32)


def phash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    transformed = DCT32 @ pixels @ DCT32.T
    low = transformed[:8, :8].copy()
    median = np.median(low.ravel()[1:])
    bits = (low >= median).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def gray_square(path, size):
    with Image.open(path) as image:
        return np.asarray(ImageOps.fit(image.convert("L"), (size, size), method=Image.Resampling.LANCZOS), dtype=np.float32)


def gaussian_ssim(a, b):
    # An 11x11 uniform local window is used here as a dependency-free SSIM
    # implementation. pHash is only a candidate generator; SSIM is decisive.
    def box_mean(array, radius=5):
        padded = np.pad(array, radius, mode="reflect")
        summed = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        width = radius * 2 + 1
        return (summed[width:, width:] - summed[:-width, width:]
                - summed[width:, :-width] + summed[:-width, :-width]) / (width * width)

    mu_a, mu_b = box_mean(a), box_mean(b)
    var_a = np.maximum(0, box_mean(a * a) - mu_a * mu_a)
    var_b = np.maximum(0, box_mean(b * b) - mu_b * mu_b)
    cov = box_mean(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(np.mean(score))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_of(path, root):
    return path.relative_to(root).parts[0]


def main():
    args = parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"输出目录已存在，停止覆盖：{output}")
    images = sorted(p for p in root.glob("*/images/*") if p.is_file())
    records = []
    gray_cache = {}
    for index, path in enumerate(images, 1):
        records.append((path, phash(path)))
        gray_cache[path] = gray_square(path, args.ssim_size)
        if index % 500 == 0:
            print(f"pHash {index}/{len(images)}", flush=True)

    max_distance = int(np.floor((1 - args.phash_similarity) * 64 + 1e-9))
    phash_candidates = []
    for (path_a, hash_a), (path_b, hash_b) in itertools.combinations(records, 2):
        distance = (hash_a ^ hash_b).bit_count()
        if distance <= max_distance:
            phash_candidates.append((path_a, path_b, distance))
    print(f"pHash candidates={len(phash_candidates)} distance<={max_distance}", flush=True)

    accepted = []
    for index, (path_a, path_b, distance) in enumerate(phash_candidates, 1):
        score = gaussian_ssim(gray_cache[path_a], gray_cache[path_b])
        if score >= args.ssim:
            accepted.append((path_a, path_b, distance, score))
        if index % 500 == 0:
            print(f"SSIM {index}/{len(phash_candidates)} accepted={len(accepted)}", flush=True)

    output.mkdir(parents=True)
    fields = ["候选编号", "图片A划分", "图片A", "图片B划分", "图片B", "是否跨划分", "pHash汉明距离",
              "pHash相似度", "SSIM", "SHA256是否相同", "人工审核结论", "拟保留图片", "备注"]
    rows = []
    for number, (a, b, distance, score) in enumerate(sorted(accepted, key=lambda x: -x[3]), 1):
        split_a, split_b = split_of(a, root), split_of(b, root)
        rows.append({"候选编号": number, "图片A划分": split_a, "图片A": str(a.relative_to(root)),
                     "图片B划分": split_b, "图片B": str(b.relative_to(root)),
                     "是否跨划分": "是" if split_a != split_b else "否", "pHash汉明距离": distance,
                     "pHash相似度": f"{1-distance/64:.6f}", "SSIM": f"{score:.6f}",
                     "SHA256是否相同": "是" if sha256(a) == sha256(b) else "否",
                     "人工审核结论": "", "拟保留图片": "", "备注": ""})
    with (output / "视觉近重复候选.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    split_pairs = Counter("-".join(sorted((r["图片A划分"], r["图片B划分"]))) for r in rows)
    summary = {"数据集": str(root), "图片数": len(images), "pHash": "32×32灰度DCT低频8×8，64位",
               "pHash阈值": args.phash_similarity, "最大汉明距离": max_distance,
               "SSIM": f"{args.ssim_size}×{args.ssim_size}灰度、11×11局部窗口", "SSIM阈值": args.ssim,
               "pHash候选对": len(phash_candidates), "通过SSIM候选对": len(rows),
               "候选划分组合": dict(sorted(split_pairs.items())), "说明": "仅生成候选，未删除图片"}
    (output / "视觉近重复分析汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
