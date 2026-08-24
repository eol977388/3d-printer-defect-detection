"""Create a dry-run representative-frame and split plan for cross-split clusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_FIXES = {"nozzle_heavy_contamination_0011": 0}
SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args():
    parser = argparse.ArgumentParser(description="生成跨划分相似簇v2拟处理方案")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--whole-ssim", type=float, default=0.95)
    parser.add_argument("--roi-ssim", type=float, default=0.95)
    parser.add_argument("--box-iou", type=float, default=0.90)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--roi-expand", type=float, default=0.20)
    return parser.parse_args()


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for row in rows:
        union(row["图片A"], row["图片B"])
    groups = defaultdict(set)
    for item in parent:
        groups[find(item)].add(item)
    cross = [members for members in groups.values() if len({x.split("/")[0] for x in members}) > 1]
    cross.sort(key=lambda members: (-len(members), sorted(members)[0]))
    return cross


def label_path(image):
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def read_one_box(path, stem):
    rows = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError(f"预期每张图恰好一个框：{path}")
    cls = int(float(rows[0][0]))
    corrected = CLASS_FIXES.get(stem, cls)
    return cls, corrected, tuple(float(x) for x in rows[0][1:])


def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    aa = (ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2)
    bb = (bx-bw/2, by-bh/2, bx+bw/2, by+bh/2)
    iw = max(0.0, min(aa[2], bb[2]) - max(aa[0], bb[0]))
    ih = max(0.0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def box_mean(array, radius=5):
    padded = np.pad(array, radius, mode="reflect")
    summed = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = radius * 2 + 1
    return (summed[width:, width:] - summed[:-width, width:]
            - summed[width:, :-width] + summed[:-width, :-width]) / (width * width)


def ssim(a, b):
    mu_a, mu_b = box_mean(a), box_mean(b)
    var_a = np.maximum(0, box_mean(a*a) - mu_a*mu_a)
    var_b = np.maximum(0, box_mean(b*b) - mu_b*mu_b)
    cov = box_mean(a*b) - mu_a*mu_b
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    score = ((2*mu_a*mu_b+c1)*(2*cov+c2))/((mu_a**2+mu_b**2+c1)*(var_a+var_b+c2))
    return float(np.mean(score))


def laplacian_variance(gray):
    inner = (-4*gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
             + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(np.var(inner))


def roi_image(image, box, size, expand):
    x, y, w, h = box
    w, h = w*(1+2*expand), h*(1+2*expand)
    left = max(0, round((x-w/2)*image.width)); right = min(image.width, round((x+w/2)*image.width))
    top = max(0, round((y-h/2)*image.height)); bottom = min(image.height, round((y+h/2)*image.height))
    if right <= left or bottom <= top:
        raise ValueError("无效标注框裁剪")
    return np.asarray(ImageOps.fit(image.crop((left, top, right, bottom)), (size, size), method=Image.Resampling.LANCZOS), dtype=np.float32)


def load_item(root, rel_text, size, expand):
    rel, path = Path(rel_text), root / rel_text
    original, corrected, box = read_one_box(label_path(path), path.stem)
    with Image.open(path) as opened:
        gray_image = opened.convert("L")
        whole = np.asarray(ImageOps.fit(gray_image, (size, size), method=Image.Resampling.LANCZOS), dtype=np.float32)
        roi = roi_image(gray_image, box, size, expand)
    brightness = float(np.mean(whole)); over = float(np.mean(whole >= 245)); under = float(np.mean(whole <= 10))
    sharpness = laplacian_variance(whole)
    # High sharpness is good; extreme brightness/exposure is penalized.
    quality = math.log1p(sharpness) - abs(brightness-128)/64 - 3*over - 2*under
    return {"rel": rel_text, "path": path, "split": rel.parts[0], "original_class": original,
            "class": corrected, "box": box, "whole": whole, "roi": roi, "brightness": brightness,
            "over": over, "under": under, "sharpness": sharpness, "quality": quality}


def count_fixed(root, clustered):
    counts = {split: Counter() for split in SPLITS}
    clustered = set(clustered)
    for label in root.glob("*/labels/*.txt"):
        rel_image = str(label.relative_to(root).with_suffix(".jpg")).replace("/labels/", "/images/")
        # Accommodate non-jpg images by stem lookup, though this dataset is jpg.
        if rel_image in clustered:
            continue
        rows = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for row in rows:
            cls = CLASS_FIXES.get(label.stem, int(float(row[0])))
            counts[label.relative_to(root).parts[0]][cls] += 1
    return counts


def assign_splits(cluster_plans, fixed):
    total_by_class = Counter()
    for split in SPLITS:
        total_by_class.update(fixed[split])
    for plan in cluster_plans:
        total_by_class[plan["class"]] += len(plan["kept"])
    desired = {split: {cls: total_by_class[cls]*TARGET[split] for cls in total_by_class} for split in SPLITS}
    current = {split: Counter(fixed[split]) for split in SPLITS}
    for plan in sorted(cluster_plans, key=lambda p: -len(p["kept"])):
        cls, amount = plan["class"], len(plan["kept"])
        def cost(split):
            before = abs(current[split][cls]-desired[split][cls])
            after = abs(current[split][cls]+amount-desired[split][cls])
            total_before = abs(sum(current[split].values())-sum(desired[split].values()))
            total_after = abs(sum(current[split].values())+amount-sum(desired[split].values()))
            return (after-before) + 0.35*(total_after-total_before)
        chosen = min(SPLITS, key=lambda split: (cost(split), SPLITS.index(split)))
        plan["target_split"] = chosen
        current[chosen][cls] += amount
    return current, desired


def main():
    args = parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"输出目录已存在，停止覆盖：{output}")
    candidate_rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    clusters = components(candidate_rows)
    all_members = sorted(set().union(*clusters))
    cache = {}
    for number, rel in enumerate(all_members, 1):
        cache[rel] = load_item(root, rel, args.size, args.roi_expand)
        if number % 200 == 0:
            print(f"Loaded {number}/{len(all_members)}", flush=True)

    cluster_plans, pair_rows = [], []
    for cluster_id, members in enumerate(clusters, 1):
        items = [cache[x] for x in members]
        classes = {item["class"] for item in items}
        if len(classes) != 1:
            raise ValueError(f"第{cluster_id}簇拟修正后仍有类别冲突：{classes}")
        redundancy = defaultdict(list)
        for i, a in enumerate(items):
            for b in items[i+1:]:
                whole_score, roi_score, iou = ssim(a["whole"], b["whole"]), ssim(a["roi"], b["roi"]), box_iou(a["box"], b["box"])
                redundant = (whole_score >= args.whole_ssim and roi_score >= args.roi_ssim and
                             iou >= args.box_iou and a["class"] == b["class"])
                pair_rows.append({"相似簇编号": cluster_id, "图片A": a["rel"], "图片B": b["rel"],
                                  "整图SSIM": f"{whole_score:.6f}", "目标区域SSIM": f"{roi_score:.6f}",
                                  "框IoU": f"{iou:.6f}", "是否高度冗余": "是" if redundant else "否"})
                if redundant:
                    redundancy[a["rel"]].append(b["rel"]); redundancy[b["rel"]].append(a["rel"])
        kept, dropped = [], {}
        for item in sorted(items, key=lambda x: (-x["quality"], x["rel"])):
            matches = [rep for rep in kept if rep["rel"] in redundancy[item["rel"]]]
            if matches:
                dropped[item["rel"]] = max(matches, key=lambda x: x["quality"])["rel"]
            else:
                kept.append(item)
        cluster_plans.append({"id": cluster_id, "members": items, "kept": kept, "dropped": dropped,
                              "class": next(iter(classes))})
        if cluster_id % 25 == 0:
            print(f"Planned {cluster_id}/{len(clusters)} clusters", flush=True)

    fixed = count_fixed(root, all_members)
    predicted, desired = assign_splits(cluster_plans, fixed)
    output.mkdir(parents=True)
    plan_fields = ["相似簇编号", "原数据划分", "图片", "原类别", "拟修正类别", "处理决定", "对应代表帧",
                   "清晰度", "平均亮度", "过曝比例", "欠曝比例", "质量分", "拟数据划分", "处理理由", "人工审核"]
    plan_rows = []
    for plan in cluster_plans:
        for item in sorted(plan["members"], key=lambda x: x["rel"]):
            is_drop = item["rel"] in plan["dropped"]
            representative = plan["dropped"].get(item["rel"], item["rel"])
            plan_rows.append({"相似簇编号": plan["id"], "原数据划分": item["split"], "图片": item["rel"],
                              "原类别": item["original_class"], "拟修正类别": item["class"],
                              "处理决定": "拟删除" if is_drop else "拟保留", "对应代表帧": representative,
                              "清晰度": f"{item['sharpness']:.6f}", "平均亮度": f"{item['brightness']:.6f}",
                              "过曝比例": f"{item['over']:.6f}", "欠曝比例": f"{item['under']:.6f}",
                              "质量分": f"{item['quality']:.6f}",
                              "拟数据划分": "删除" if is_drop else plan["target_split"],
                              "处理理由": "与质量更好的代表帧同时满足整图、目标区域和框IoU阈值" if is_drop else "保留有效视觉差异或作为高质量代表帧",
                              "人工审核": ""})
    with (output / "抽帧拟处理清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=plan_fields); writer.writeheader(); writer.writerows(plan_rows)
    with (output / "簇内两两指标.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    split_rows = []
    for plan in cluster_plans:
        split_rows.append({"相似簇编号": plan["id"], "原图片数": len(plan["members"]), "拟保留数": len(plan["kept"]),
                           "拟删除数": len(plan["dropped"]), "类别": plan["class"], "原划分": ";".join(sorted({x['split'] for x in plan['members']})),
                           "拟统一划分": plan["target_split"]})
    with (output / "相似簇重新划分清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0])); writer.writeheader(); writer.writerows(split_rows)
    fix_rows = [{"图片": stem+".jpg", "原类别": 1, "拟修正类别": cls, "人工结论": "裹头", "状态": "仅列入预案，尚未修改v1"}
                for stem, cls in CLASS_FIXES.items()]
    with (output / "第5项人工标签修正.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fix_rows[0])); writer.writeheader(); writer.writerows(fix_rows)
    kept_count = sum(len(x["kept"]) for x in cluster_plans)
    summary = {"数据集": str(root), "模式": "dry-run，未修改数据集", "跨划分相似簇": len(cluster_plans),
               "涉及图片": len(all_members), "拟保留代表帧": kept_count, "拟删除高度冗余帧": len(all_members)-kept_count,
               "阈值": {"整图SSIM": args.whole_ssim, "目标区域SSIM": args.roi_ssim, "标注框IoU": args.box_iou,
                        "目标框扩展比例": args.roi_expand},
               "处理后预计类别与划分": {split: {str(k): v for k, v in sorted(predicted[split].items())} for split in SPLITS},
               "目标比例": TARGET, "人工标签拟修正": CLASS_FIXES}
    (output / "抽帧方案汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = ["# 跨划分相似簇抽帧与重划分预案", "", "本报告为预案，没有修改或删除clean_v1数据。", "",
                f"- 跨划分相似簇：{len(cluster_plans)}个", f"- 涉及图片：{len(all_members)}张",
                f"- 拟保留：{kept_count}张", f"- 拟删除：{len(all_members)-kept_count}张", "",
                "判定为高度冗余必须同时满足：整图SSIM≥0.95、目标区域SSIM≥0.95、框IoU≥0.90、类别一致。",
                "同一原始相似簇的代表帧拟统一分配到同一个数据划分。"]
    (output / "抽帧方案汇总.md").write_text("\n".join(markdown)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
