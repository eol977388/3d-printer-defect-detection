"""Create deterministic annotated samples for every filename-prefix category."""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples-per-category", type=int, default=30)
    p.add_argument("--max-height", type=int, default=850)
    return p.parse_args()


def font(size, bold=False):
    paths = [Path("/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc"),
             Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def prefix(stem):
    return re.sub(r"_\d+$", "", stem)


def label_path(image):
    return Path(str(image).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def labels(path):
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            values = line.split()
            result.append((int(float(values[0])), *(float(v) for v in values[1:])))
    return result


def select_evenly(items, count):
    if len(items) <= count:
        return items
    indexes = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indexes]


def render(image_path, split, category, output):
    source = Image.open(image_path).convert("RGB")
    scale = min(1.0, 850 / source.height)
    display = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
    rows = labels(label_path(image_path))
    header = 112
    canvas = Image.new("RGB", (display.width, display.height + header), "#202124")
    canvas.paste(display, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f"{category}｜{split}｜{image_path.name}", fill="white", font=font(25, True))
    draw.text((16, 49), f"原始标签：{rows}", fill="#E0E0E0", font=font(18))
    draw.text((16, 78), "红框=类别0（裹头）　青框=类别1（非裹头）", fill="#E0E0E0", font=font(18))
    image_draw = ImageDraw.Draw(canvas)
    colors = {0: "#E53935", 1: "#00ACC1"}
    for index, (cls, x, y, w, h) in enumerate(rows, 1):
        x1, y1 = (x-w/2)*display.width, header+(y-h/2)*display.height
        x2, y2 = (x+w/2)*display.width, header+(y+h/2)*display.height
        color = colors.get(cls, "#FFB300")
        image_draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        title = f"类别{cls} 框{index}"
        fnt = font(20, True)
        box = image_draw.textbbox((x1, max(header, y1-28)), title, font=fnt)
        image_draw.rectangle(box, fill=color)
        image_draw.text((x1, max(header, y1-28)), title, fill="white", font=fnt)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=91, optimize=True)
    return rows


def main():
    a = args()
    grouped = defaultdict(list)
    for split in ("train", "val", "test"):
        root = a.dataset / split / "images"
        for image in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            grouped[prefix(image.stem)].append((split, image))

    images_dir = a.output / "samples"
    images_dir.mkdir(parents=True, exist_ok=False)
    manifest, sections = [], []
    for category in sorted(grouped):
        items = sorted(grouped[category], key=lambda x: (x[0], x[1].name))
        selected = select_evenly(items, a.samples_per_category)
        cards = []
        for number, (split, image) in enumerate(selected, 1):
            filename = f"{category}__{number:03d}__{split}__{image.stem}.jpg"
            rows = render(image, split, category, images_dir / filename)
            manifest.append({"文件名前缀": category, "样例序号": number, "数据划分": split,
                             "原图片": str(image), "原标签": str(label_path(image)),
                             "标签内容": repr(rows), "标注样例图": f"samples/{filename}", "审核备注": ""})
            cards.append(f'<article><h3>{number:03d}｜{html.escape(split)}｜{html.escape(image.name)}</h3>'
                         f'<img loading="lazy" src="samples/{html.escape(filename)}"></article>')
        sections.append(f'<section id="{html.escape(category)}"><h2>{html.escape(category)}：'
                        f'展示 {len(selected)} / 总计 {len(items)}</h2>{"".join(cards)}</section>')

    fields = ["文件名前缀", "样例序号", "数据划分", "原图片", "原标签", "标签内容", "标注样例图", "审核备注"]
    with (a.output / "样例清单.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest)
    nav = "　".join(f'<a href="#{html.escape(x)}">{html.escape(x)}</a>' for x in sorted(grouped))
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>各类别标注框样例</title>
<style>body{{font-family:"Microsoft YaHei";background:#f4f5f7;max-width:1400px;margin:auto;padding:22px}}
nav{{position:sticky;top:0;background:#fff;padding:14px;z-index:3}}section{{background:white;padding:18px;margin:24px 0;border-radius:10px}}
article{{margin:25px 0;border-top:1px solid #ddd;padding-top:12px}}img{{display:block;max-width:900px;width:100%;height:auto}}
a{{color:#1565c0}}</style></head><body><h1>dataset_2各文件类别的原始标注框样例</h1>
<p>大类别等距抽取30张；不足30张的类别全部展示。红框=类别0，青框=类别1。</p><nav>{nav}</nav>{''.join(sections)}</body></html>'''
    (a.output / "index.html").write_text(page, encoding="utf-8")
    (a.output / "说明.txt").write_text(
        "本目录用于观察各文件名前缀的原始标框习惯。\n大类别等距抽取30张，少于30张的类别全部展示。\n"
        "红框表示类别0（裹头），青框表示类别1（非裹头）。\n未修改任何原始数据。\n", encoding="utf-8")
    print(f"categories={len(grouped)} samples={len(manifest)} output={a.output}")


if __name__ == "__main__":
    main()
