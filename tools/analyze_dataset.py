"""Read-only YOLO dataset analysis matching the 3D-printer experiment document."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIZE_BINS = ("<5", "5-16", "16-32", "32-64", "64-128", ">=128")
ZH_HEADERS = {
    "split": "数据划分", "cluster": "聚类编号", "width_norm": "归一化宽度",
    "height_norm": "归一化高度", "aspect_ratio": "宽高比", "count": "数量",
    "image_path": "图片路径", "label_path": "标签路径", "filename": "文件名",
    "prefix": "文件名前缀", "width": "图像宽度", "height": "图像高度",
    "boxes": "目标框数量", "mean_brightness": "平均亮度", "brightness_std": "亮度标准差",
    "overexposed_ratio": "过曝像素比例", "underexposed_ratio": "欠曝像素比例",
    "blur_score": "模糊度分数", "mean_r": "红通道均值", "mean_g": "绿通道均值",
    "mean_b": "蓝通道均值", "sha256": "SHA256", "class_id": "类别编号",
    "class_name": "类别名称", "box_count": "目标框数量", "percentage": "占比",
    "group_id": "重复组编号", "cross_split": "是否跨集合",
    "group_splits": "重复组所在集合", "line_number": "标签行号", "issue": "问题类型",
    "details": "问题详情", "x_center": "中心点X", "y_center": "中心点Y",
    "width_px": "像素宽度", "height_px": "像素高度", "area_px": "像素面积",
    "area_norm": "归一化面积", "size_category": "尺寸类别", "seed": "随机种子",
    "filename_prefix": "文件名前缀", "current_class": "当前类别",
    "suspected_issue": "疑似问题", "evidence": "判断依据", "review_status": "审核状态",
    "final_action": "最终处理", "reviewer": "审核人", "comment": "备注",
}


def arguments():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Analyze a YOLO detection dataset without modifying it")
    p.add_argument("--dataset", type=Path, default=root.parent / "datasets" / "dataset_2")
    p.add_argument("--output", type=Path, default=root / "reports" / "data_analysis" / "raw_v1")
    p.add_argument("--class-names", nargs="*", default=["wrapped", "not_wrapped"])
    p.add_argument("--hist-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260822)
    return p.parse_args()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([ZH_HEADERS.get(field, field) for field in fields])
        for row in rows:
            w.writerow([row.get(field, "") for field in fields])


def finish(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def stats(values):
    if not values:
        return {k: None for k in ("min", "p05", "p25", "median", "mean", "p75", "p95", "max")}
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blur_score(gray):
    if min(gray.shape) < 3:
        return 0.0
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def size_bin(w, h, image_width, image_height, analysis_size=640):
    # Match the experiment document: letterbox the longest image side to 640,
    # then classify by the square root of the scaled bounding-box area.
    scale = analysis_size / max(image_width, image_height)
    n = math.sqrt(w * h) * scale
    if n < 5: return "<5"
    if n < 16: return "5-16"
    if n < 32: return "16-32"
    if n < 64: return "32-64"
    if n < 128: return "64-128"
    return ">=128"


def parse_label(path):
    boxes, issues = [], []
    if not path.exists():
        return boxes, issues
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            issues.append((line_no, "invalid_column_count", f"columns={len(parts)}"))
            continue
        try:
            c = float(parts[0]); xywh = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append((line_no, "non_numeric_label", raw))
            continue
        if not c.is_integer():
            issues.append((line_no, "non_integer_class", parts[0]))
            continue
        c = int(c); x, y, w, h = xywh
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues.append((line_no, "coordinate_out_of_range", raw))
        if x-w/2 < 0 or x+w/2 > 1 or y-h/2 < 0 or y+h/2 > 1:
            issues.append((line_no, "box_crosses_boundary", raw))
        boxes.append((c, x, y, w, h))
    return boxes, issues


def analyze_split(dataset, split, names):
    image_dir, label_dir = dataset/split/"images", dataset/split/"labels"
    paths = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    images, boxes, issues, used_labels = [], [], [], set()
    for image_path in paths:
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if label_path.exists(): used_labels.add(label_path)
        else: issues.append([split, str(image_path), str(label_path), "", "missing_label", ""])
        labels, label_issues = parse_label(label_path)
        for n, kind, detail in label_issues:
            issues.append([split, str(image_path), str(label_path), n, kind, detail])
        try:
            with Image.open(image_path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                width, height = im.size
        except (OSError, UnidentifiedImageError, ValueError) as e:
            issues.append([split, str(image_path), str(label_path), "", "unreadable_image", repr(e)])
            continue
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        means = rgb.reshape(-1, 3).mean(0)
        prefix = re.sub(r"_\d+$", "", image_path.stem)
        images.append({"split": split, "image_path": str(image_path), "label_path": str(label_path),
                       "filename": image_path.name, "prefix": prefix, "width": width, "height": height,
                       "boxes": len(labels), "mean_brightness": float(gray.mean()),
                       "brightness_std": float(gray.std()), "overexposed_ratio": float((gray >= 245).mean()),
                       "underexposed_ratio": float((gray <= 10).mean()), "blur_score": blur_score(gray),
                       "mean_r": float(means[0]), "mean_g": float(means[1]), "mean_b": float(means[2]),
                       "sha256": file_hash(image_path)})
        duplicates = Counter(labels)
        for value, count in duplicates.items():
            if count > 1:
                issues.append([split, str(image_path), str(label_path), "", "duplicate_box", f"{value}, count={count}"])
        for line_no, (c, x, y, w, h) in enumerate(labels, 1):
            if c < 0 or c >= len(names):
                issues.append([split, str(image_path), str(label_path), line_no, "class_out_of_range", str(c)])
            wp, hp = w*width, h*height
            boxes.append({"split": split, "image_path": str(image_path), "label_path": str(label_path),
                          "line_number": line_no, "prefix": prefix, "class_id": c, "x_center": x,
                          "y_center": y, "width_norm": w, "height_norm": h, "width_px": wp,
                          "height_px": hp, "area_px": wp*hp, "area_norm": w*h,
                          # The source document defines ratio directly in normalized YOLO coordinates.
                          "aspect_ratio": w/h if h else float("inf"),
                          "size_category": size_bin(wp, hp, width, height)})
    if label_dir.exists():
        for path in sorted(set(label_dir.rglob("*.txt"))-used_labels):
            issues.append([split, "", str(path), "", "orphan_label", "no matching image"])
    return images, boxes, issues


def kmeans_anchors(boxes, k, seed):
    pts = np.array([[b["width_norm"], b["height_norm"]] for b in boxes], dtype=float)
    if not len(pts): return np.empty((0, 2)), np.empty(0, dtype=int)
    k = min(k, len(pts)); rng = np.random.default_rng(seed)
    centers = pts[rng.choice(len(pts), k, replace=False)].copy(); labels = np.zeros(len(pts), int)
    for _ in range(100):
        inter = np.minimum(pts[:, None, 0], centers[:, 0]) * np.minimum(pts[:, None, 1], centers[:, 1])
        union = pts[:, None, 0]*pts[:, None, 1] + centers[:, 0]*centers[:, 1] - inter
        new_labels = (1-inter/np.maximum(union, 1e-12)).argmin(1)
        new_centers = centers.copy()
        for i in range(k):
            selected = pts[new_labels == i]
            if len(selected): new_centers[i] = np.median(selected, axis=0)
        if np.allclose(new_centers, centers): labels = new_labels; break
        centers, labels = new_centers, new_labels
    order = np.argsort(centers.prod(1)); remap = np.zeros(k, int); remap[order] = np.arange(k)
    return centers[order], remap[labels]


def plot_classes(boxes, names, split, path):
    count = Counter(b["class_id"] for b in boxes); ids = sorted(set(range(len(names))) | set(count))
    values = [count[i] for i in ids]; labels = [f"{i}: {names[i] if i < len(names) else 'unknown'}" for i in ids]
    plt.figure(figsize=(9, 5)); bars = plt.bar(labels, values, color="#87CEEB")
    for b, v in zip(bars, values): plt.text(b.get_x()+b.get_width()/2, v, str(v), ha="center", va="bottom")
    plt.title(f"YOLO Label Distribution - {split}"); plt.ylabel("Bounding-box count"); plt.grid(axis="y", alpha=.25); finish(path)


def plot_sizes(boxes, split, path):
    c = Counter(b["size_category"] for b in boxes); values = [c[x] for x in SIZE_BINS]
    plt.figure(figsize=(10, 5)); bars = plt.bar(SIZE_BINS, values, color="#87CEEB")
    for b, v in zip(bars, values): plt.text(b.get_x()+b.get_width()/2, v, str(v), ha="center", va="bottom")
    plt.title(f"Object Size Distribution - {split}"); plt.xlabel("Equivalent box size sqrt(area), letterboxed to 640 (pixels)")
    plt.ylabel("Frequency"); plt.grid(axis="y", alpha=.25); finish(path)


def plot_shapes(boxes, split, centers, path):
    ratio = [b["aspect_ratio"] for b in boxes]; widths = [b["width_norm"] for b in boxes]; heights = [b["height_norm"] for b in boxes]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].hist(ratio, 40, color="blue", alpha=.65); ax[0,0].set_title("Bounding-box Aspect Ratio"); ax[0,0].set_xlabel("width / height")
    ax[0,1].hist(widths, 40, color="green", alpha=.65); ax[0,1].set_title("Normalized Box Width")
    ax[1,0].hist(heights, 40, color="red", alpha=.65); ax[1,0].set_title("Normalized Box Height")
    ax[1,1].scatter(widths, heights, s=8, alpha=.35, color="purple"); ax[1,1].set_title("Normalized Width-Height Scatter")
    if len(centers): ax[1,1].scatter(centers[:,0], centers[:,1], marker="x", s=100, color="red", label="K-means anchors"); ax[1,1].legend()
    for a in ax.ravel(): a.grid(alpha=.25)
    fig.suptitle(f"Bounding-box Shape Distribution - {split}"); finish(path)


def plot_rgb(images, split, samples, seed, path):
    rng = random.Random(seed + sum(map(ord, split))); chosen = images if len(images) <= samples else rng.sample(images, samples)
    hist = np.zeros((3,256), dtype=np.int64)
    for r in chosen:
        with Image.open(r["image_path"]) as im: rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        for i in range(3): hist[i] += np.bincount(rgb[...,i].ravel(), minlength=256)
    plt.figure(figsize=(11,6)); x=np.arange(256)
    for i,(name,color) in enumerate((("Red","red"),("Green","green"),("Blue","blue"))): plt.plot(x,hist[i],color=color,label=f"{name} channel")
    plt.title(f"Color Histogram Distribution ({len(chosen)} deterministic images) - {split}"); plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency"); plt.legend(); plt.grid(alpha=.25); finish(path)
    return [{"split":split,"image_path":r["image_path"],"seed":seed} for r in chosen]


def plot_quality(images, out):
    for filename, title, key, xlabel in (
        ("09_brightness_distribution.png","Brightness","mean_brightness","Mean grayscale brightness"),
        ("10_overexposure_ratio_distribution.png","Overexposure","overexposed_ratio","Ratio of pixels >= 245"),
        ("11_blur_score_distribution.png","Blur score","blur_score","Laplacian variance (higher is sharper)")):
        train = np.asarray([r[key] for r in images if r["split"] == "train"], dtype=float)
        val = np.asarray([r[key] for r in images if r["split"] == "val"], dtype=float)
        combined = np.concatenate((train, val))
        bins = np.linspace(combined.min(), combined.max(), 41)
        plt.figure(figsize=(9,5))
        # Normalize each split to 100% so the smaller val set remains directly comparable.
        plt.hist(train, bins=bins, weights=np.full(len(train), 100 / len(train)),
                 color="#5B9BD5", alpha=.55, edgecolor="#2F5597", label=f"Train (n={len(train)})")
        plt.hist(val, bins=bins, weights=np.full(len(val), 100 / len(val)),
                 color="#ED7D31", alpha=.55, edgecolor="#C65911", label=f"Val (n={len(val)})")
        plt.title(f"Image {title} Distribution: Train vs Val"); plt.xlabel(xlabel)
        plt.ylabel("Percentage of split images (%)"); plt.legend(); plt.grid(axis="y",alpha=.25)
        finish(out/"figures"/filename)


def main():
    a=arguments(); dataset=a.dataset.resolve(); out=a.output.resolve()
    if not dataset.exists(): raise SystemExit(f"Dataset not found: {dataset}")
    (out/"tables").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True)
    all_images=[]; all_boxes=[]; all_issues=[]; summary={}; anchors=[]; samples=[]
    for split in SPLITS:
        images,boxes,issues=analyze_split(dataset,split,a.class_names); all_images+=images; all_boxes+=boxes; all_issues+=issues
        cc=Counter(b["class_id"] for b in boxes); sc=Counter(b["size_category"] for b in boxes)
        summary[split]={"images":len(images),"boxes":len(boxes),"issues":len(issues),"class_counts":dict(sorted(cc.items())),
                        "size_counts":{x:sc[x] for x in SIZE_BINS},"aspect_ratio":stats([b["aspect_ratio"] for b in boxes]),
                        "width_norm":stats([b["width_norm"] for b in boxes]),"height_norm":stats([b["height_norm"] for b in boxes])}
        if split in ("train","val"):
            plot_classes(boxes,a.class_names,split,out/"figures"/("01_train_class_distribution.png" if split=="train" else "02_val_class_distribution.png"))
            plot_sizes(boxes,split,out/"figures"/("03_train_object_size_distribution.png" if split=="train" else "04_val_object_size_distribution.png"))
            centers,labels=kmeans_anchors(boxes,9,a.seed); plot_shapes(boxes,split,centers,out/"figures"/("05_train_bbox_shape_distribution.png" if split=="train" else "06_val_bbox_shape_distribution.png"))
            counts=Counter(labels.tolist())
            for i,c in enumerate(centers): anchors.append({"split":split,"cluster":i+1,"width_norm":c[0],"height_norm":c[1],"aspect_ratio":c[0]/c[1],"count":counts[i]})
            samples += plot_rgb(images,split,a.hist_samples,a.seed,out/"figures"/("07_train_rgb_histogram.png" if split=="train" else "08_val_rgb_histogram.png"))
    plot_quality([x for x in all_images if x["split"] in ("train","val")],out)
    image_fields=list(all_images[0]); box_fields=list(all_boxes[0]); issue_fields=["split","image_path","label_path","line_number","issue","details"]
    write_csv(out/"tables"/"image_quality_metrics.csv",all_images,image_fields); write_csv(out/"tables"/"object_size_details.csv",all_boxes,box_fields)
    write_csv(out/"tables"/"integrity_issues.csv",[dict(zip(issue_fields,x)) for x in all_issues],issue_fields)
    write_csv(out/"tables"/"anchor_clusters.csv",anchors,["split","cluster","width_norm","height_norm","aspect_ratio","count"])
    write_csv(out/"tables"/"rgb_histogram_sample_manifest.csv",samples,["split","image_path","seed"])
    class_rows=[]; size_rows=[]
    for split,d in summary.items():
        for c,n in d["class_counts"].items(): class_rows.append({"split":split,"class_id":c,"class_name":a.class_names[int(c)] if int(c)<len(a.class_names) else "unknown","box_count":n,"percentage":n/d["boxes"] if d["boxes"] else 0})
        for k,n in d["size_counts"].items(): size_rows.append({"split":split,"size_category":k,"box_count":n,"percentage":n/d["boxes"] if d["boxes"] else 0})
    write_csv(out/"tables"/"class_distribution.csv",class_rows,["split","class_id","class_name","box_count","percentage"])
    write_csv(out/"tables"/"object_size_distribution.csv",size_rows,["split","size_category","box_count","percentage"])
    groups=defaultdict(list)
    for r in all_images: groups[r["sha256"]].append(r)
    duplicate_rows=[]; gid=0
    for digest,records in groups.items():
        if len(records)<2: continue
        gid+=1; splits=sorted({r["split"] for r in records})
        for r in records: duplicate_rows.append({"group_id":gid,"sha256":digest,"split":r["split"],"image_path":r["image_path"],"cross_split":len(splits)>1,"group_splits":",".join(splits)})
    write_csv(out/"tables"/"exact_duplicate_images.csv",duplicate_rows,["group_id","sha256","split","image_path","cross_split","group_splits"])
    prefix_counts=defaultdict(Counter)
    for b in all_boxes: prefix_counts[b["prefix"]][b["class_id"]]+=1
    suspects=[]
    for b in all_boxes:
        majority,n=prefix_counts[b["prefix"]].most_common(1)[0]; total=sum(prefix_counts[b["prefix"]].values())
        if total>=5 and n/total>=.8 and b["class_id"]!=majority:
            suspects.append({"split":b["split"],"image_path":b["image_path"],"label_path":b["label_path"],"line_number":b["line_number"],"filename_prefix":b["prefix"],"current_class":b["class_id"],"suspected_issue":"prefix_majority_class_mismatch","evidence":f"majority={majority}, {n}/{total}","review_status":"pending","final_action":"","reviewer":"","comment":""})
    for x in all_issues:
        suspects.append({"split":x[0],"image_path":x[1],"label_path":x[2],"line_number":x[3],"filename_prefix":"","current_class":"","suspected_issue":x[4],"evidence":x[5],"review_status":"pending","final_action":"","reviewer":"","comment":""})
    suspect_fields=["split","image_path","label_path","line_number","filename_prefix","current_class","suspected_issue","evidence","review_status","final_action","reviewer","comment"]
    write_csv(out/"suspect_labels.csv",suspects,suspect_fields)
    quality=[r for r in all_images if r["split"] in ("train","val")]
    write_csv(out/"tables"/"overexposed_images.csv",sorted(quality,key=lambda x:x["overexposed_ratio"],reverse=True)[:200],image_fields)
    write_csv(out/"tables"/"blurred_images.csv",sorted(quality,key=lambda x:x["blur_score"])[:200],image_fields)
    result={"dataset":str(dataset),"seed":a.seed,"class_names":a.class_names,"splits":summary,"total_images":len(all_images),"total_boxes":len(all_boxes),
            "integrity_issue_count":len(all_issues),"suspect_count":len(suspects),"exact_duplicate_groups":gid,
            "cross_split_duplicate_groups":len({r["group_id"] for r in duplicate_rows if r["cross_split"]}),
            "quality_train_val":{"brightness":stats([r["mean_brightness"] for r in quality]),"overexposure_ratio":stats([r["overexposed_ratio"] for r in quality]),"blur_score":stats([r["blur_score"] for r in quality])}}
    (out/"dataset_analysis_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Dataset analysis report — raw_v1","","> Read-only analysis; no source image or label was modified.","",f"- Dataset: `{dataset}`",f"- Images: {len(all_images)}",f"- Boxes: {len(all_boxes)}",f"- Suspect review rows: {len(suspects)}",f"- Exact duplicate groups: {gid}",f"- Cross-split duplicate groups: {result['cross_split_duplicate_groups']}","","| Split | Images | Boxes | Classes |","|---|---:|---:|---|"]
    for split in SPLITS: lines.append(f"| {split} | {summary[split]['images']} | {summary[split]['boxes']} | {summary[split]['class_counts']} |")
    lines += ["","## Key findings"]
    for split in ("train","val"):
        d=summary[split]; r=d["aspect_ratio"]
        lines += ["",f"### {split}",f"- Aspect ratio: min={r['min']:.4f}, mean={r['mean']:.4f}, max={r['max']:.4f}.",f"- Normalized width: {d['width_norm']['min']:.4f}–{d['width_norm']['max']:.4f}.",f"- Normalized height: {d['height_norm']['min']:.4f}–{d['height_norm']['max']:.4f}.",f"- Size bins: {d['size_counts']}." ]
    lines += ["","## Rules","","- Filename/class mismatches are review candidates, never automatic relabels.","- Test is checked for integrity/leakage but must not drive model tuning.","- K-means anchors apply to YOLOv5; YOLOv8 is anchor-free."]
    (out/"dataset_analysis_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
