"""Generate side-by-side review images for exact duplicates with different YOLO boxes.

This tool is read-only with respect to the source dataset. It consumes the exact
duplicate manifest created by analyze_dataset.py and writes review artifacts to
a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render exact-duplicate label-box comparisons")
    parser.add_argument(
        "--duplicates-csv",
        type=Path,
        default=project / "reports/data_analysis/raw_v1/tables/exact_duplicate_images.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop/3D打印项目sop/916组重复图片标注框对比",
    )
    parser.add_argument("--max-panel-height", type=int, default=820)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def load_font(size, bold=False):
    candidates = [
        Path("/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def label_path(image_path):
    return Path(str(image_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def read_yolo(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_no}: {line}")
        rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
    return rows


def xyxy(box, width, height):
    _, x, y, w, h = box
    return ((x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height)


def box_iou(a, b):
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    aa = (ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2)
    bb = (bx-bw/2, by-bh/2, bx+bw/2, by+bh/2)
    iw = max(0.0, min(aa[2], bb[2]) - max(aa[0], bb[0]))
    ih = max(0.0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def fit_image(image, max_height):
    scale = min(1.0, max_height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def draw_boxes(panel, boxes, color, width=5):
    draw = ImageDraw.Draw(panel)
    font = load_font(22, bold=True)
    for index, box in enumerate(boxes, 1):
        coords = xyxy(box, panel.width, panel.height)
        draw.rectangle(coords, outline=color, width=width)
        label = f"类别 {box[0]}  框 {index}"
        left, top = coords[0], max(0, coords[1] - 31)
        text_box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((left, top), label, fill="white", font=font)


def render_group(group_id, first, second, out_path, max_height, quality):
    path_a, path_b = Path(first[3]), Path(second[3])
    label_a, label_b = label_path(path_a), label_path(path_b)
    boxes_a, boxes_b = read_yolo(label_a), read_yolo(label_b)
    image_a = Image.open(path_a).convert("RGB")
    image_b = Image.open(path_b).convert("RGB")
    panel_a, _ = fit_image(image_a, max_height)
    panel_b, _ = fit_image(image_b, max_height)
    if panel_a.size != panel_b.size:
        common = (min(panel_a.width, panel_b.width), min(panel_a.height, panel_b.height))
        panel_a = panel_a.resize(common, Image.Resampling.LANCZOS)
        panel_b = panel_b.resize(common, Image.Resampling.LANCZOS)
    draw_boxes(panel_a, boxes_a, "#E53935")
    draw_boxes(panel_b, boxes_b, "#00ACC1")

    header, gap, margin = 164, 24, 18
    canvas_w = margin * 2 + panel_a.width * 2 + gap
    canvas_h = header + panel_a.height + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#202124")
    canvas.paste(panel_a, (margin, header))
    canvas.paste(panel_b, (margin + panel_a.width + gap, header))
    draw = ImageDraw.Draw(canvas)
    title_font, body_font = load_font(28, True), load_font(20)
    iou = box_iou(boxes_a[0], boxes_b[0]) if len(boxes_a) == len(boxes_b) == 1 else float("nan")
    draw.text((margin, 12), f"重复组 {int(group_id):04d}　同一张图片的两套标注框　IoU={iou:.4f}", fill="white", font=title_font)
    draw.text((margin, 58), f"红框｜{first[2]}｜{path_a.name}", fill="#FF8A80", font=body_font)
    draw.text((margin, 91), f"标签：{boxes_a}", fill="#FFCDD2", font=body_font)
    x2 = margin + panel_a.width + gap
    draw.text((x2, 58), f"青框｜{second[2]}｜{path_b.name}", fill="#80DEEA", font=body_font)
    draw.text((x2, 91), f"标签：{boxes_b}", fill="#B2EBF2", font=body_font)
    draw.text((margin, 128), "人工审核：比较两套框的位置和范围；本图不修改原始图片或标签。", fill="#E0E0E0", font=body_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=quality, optimize=True)
    return path_a, label_a, boxes_a, path_b, label_b, boxes_b, iou


def main():
    args = parse_args()
    rows = list(csv.reader(args.duplicates_csv.open(encoding="utf-8-sig")))
    groups = {}
    # Header may be Chinese or English. Data column positions are stable:
    # group id, hash, split, image path, cross-split flag, group splits.
    for row in rows[1:]:
        groups.setdefault(row[0], []).append(row)
    invalid = {group: items for group, items in groups.items() if len(items) != 2}
    if invalid:
        raise SystemExit(f"Expected exactly 2 images per group, invalid groups: {list(invalid)[:10]}")

    comparisons = args.output / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=False)
    manifest = []
    html_cards = []
    for number, group_id in enumerate(sorted(groups, key=lambda value: int(value)), 1):
        first, second = sorted(groups[group_id], key=lambda row: (row[2], row[3]))
        stem_a = Path(first[3]).stem
        stem_b = Path(second[3]).stem
        filename = f"group_{int(group_id):04d}__{stem_a}__VS__{stem_b}.jpg"
        result = render_group(group_id, first, second, comparisons / filename,
                              args.max_panel_height, args.jpeg_quality)
        path_a, label_a, boxes_a, path_b, label_b, boxes_b, iou = result
        manifest.append({
            "重复组编号": int(group_id), "对比图": f"comparisons/{filename}",
            "图片A": str(path_a), "标签A": str(label_a), "标注框A": repr(boxes_a),
            "图片B": str(path_b), "标签B": str(label_b), "标注框B": repr(boxes_b),
            "框IoU": f"{iou:.6f}", "审核结论": "", "采用标签": "", "备注": "",
        })
        html_cards.append(
            f'<section id="group-{int(group_id)}"><h2>重复组 {int(group_id):04d}　IoU={iou:.4f}</h2>'
            f'<img loading="lazy" src="comparisons/{html.escape(filename)}" alt="重复组 {int(group_id)}"></section>'
        )
        if number % 100 == 0:
            print(f"Rendered {number}/{len(groups)}")

    fields = ["重复组编号", "对比图", "图片A", "标签A", "标注框A", "图片B", "标签B", "标注框B",
              "框IoU", "审核结论", "采用标签", "备注"]
    with (args.output / "审核清单.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>916组重复图片标注框对比</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;background:#f4f5f7;color:#202124;margin:0 auto;max-width:1500px;padding:24px}}
h1{{position:sticky;top:0;background:#f4f5f7;padding:16px 0;z-index:2}}section{{background:white;margin:24px 0;padding:18px;border-radius:10px;box-shadow:0 2px 8px #0002}}
img{{display:block;width:100%;height:auto}}h2{{margin-top:0}}code{{background:#eee;padding:2px 5px}}
</style></head><body><h1>916组完全重复图片的标注框对比</h1>
<p>红框为图片A标签，青框为图片B标签。请把审核结果填写到 <code>审核清单.csv</code>。</p>
{''.join(html_cards)}</body></html>'''
    (args.output / "index.html").write_text(page, encoding="utf-8")
    (args.output / "说明.txt").write_text(
        "本目录包含916张左右对比图。\n红框：第一份标签；青框：第二份标签。\n"
        "请双击index.html连续浏览，或在comparisons目录逐张查看。\n"
        "审核清单.csv预留了审核结论、采用标签和备注列。\n"
        "本工具没有修改任何原始图片和标签。\n", encoding="utf-8")
    print(f"Done: {len(groups)} groups -> {args.output}")


if __name__ == "__main__":
    main()
