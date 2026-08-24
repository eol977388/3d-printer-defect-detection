"""Render annotated review galleries for cross-split visual-similarity clusters."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="生成跨划分视觉相似簇审核图")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-height", type=int, default=620)
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def font(size, bold=False):
    paths = [Path(f"/mnt/c/Windows/Fonts/{'msyhbd' if bold else 'msyh'}.ttc")]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def label_path(image):
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def read_boxes(path):
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            values = line.split()
            boxes.append((int(float(values[0])), *(float(x) for x in values[1:])))
    return boxes


def render(source, destination, relative, max_height, quality):
    image = Image.open(source).convert("RGB")
    scale = min(1.0, max_height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    header = 104
    canvas = Image.new("RGB", (image.width, image.height + header), "#202124")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    boxes = read_boxes(label_path(source))
    for number, (cls, x, y, w, h) in enumerate(boxes, 1):
        color = "#E53935" if cls == 0 else "#00ACC1"
        coords = ((x-w/2)*image.width, header+(y-h/2)*image.height,
                  (x+w/2)*image.width, header+(y+h/2)*image.height)
        draw.rectangle(coords, outline=color, width=5)
        label = f"类别{cls} 框{number}"
        box = draw.textbbox((coords[0], max(header, coords[1]-29)), label, font=font(20, True))
        draw.rectangle(box, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font(20, True))
    split = relative.parts[0]
    draw.text((12, 10), source.name, fill="white", font=font(23, True))
    draw.text((12, 48), f"划分：{split}　标签：{boxes}", fill="#E0E0E0", font=font(18))
    draw.text((12, 76), "红框=类别0（裹头）　青框=类别1（非裹头）", fill="#E0E0E0", font=font(17))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=quality, optimize=True)
    return boxes


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
    return groups, find


def main():
    args = parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"输出目录已存在，停止覆盖：{output}")
    rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    groups, find = components(rows)
    cross = [members for members in groups.values() if len({x.split("/")[0] for x in members}) > 1]
    cross.sort(key=lambda members: (-len(members), sorted(members)[0]))
    output.mkdir(parents=True)
    manifest, sections, conflict_id = [], [], None
    for cluster_number, members in enumerate(cross, 1):
        cluster_id = f"cluster_{cluster_number:03d}"
        member_cards, classes = [], set()
        for member_number, rel_text in enumerate(sorted(members), 1):
            rel = Path(rel_text)
            source = root / rel
            filename = f"{member_number:03d}__{rel.parts[0]}__{source.name}"
            destination = output / "clusters" / cluster_id / filename
            boxes = render(source, destination, rel, args.max_height, args.quality)
            classes.update(box[0] for box in boxes)
            shown = destination.relative_to(output).as_posix()
            member_cards.append(f'<figure><img loading="lazy" src="{html.escape(shown)}"><figcaption>{html.escape(rel_text)}</figcaption></figure>')
            manifest.append({"相似簇编号": cluster_number, "簇内序号": member_number, "图片": rel_text,
                             "数据划分": rel.parts[0], "二分类标签": ";".join(str(x[0]) for x in boxes),
                             "审核图": shown, "人工结论": "", "是否保留": "", "备注": ""})
        edges = [r for r in rows if r["图片A"] in members and r["图片B"] in members]
        edge_html = "".join(f'<tr><td>{html.escape(r["图片A"])}</td><td>{html.escape(r["图片B"])}</td>'
                            f'<td>{r["pHash相似度"]}</td><td>{r["SSIM"]}</td></tr>' for r in edges)
        conflict = len(classes) > 1
        if conflict:
            conflict_id = cluster_id
        sections.append(f'<section id="{cluster_id}"><h2>相似簇 {cluster_number:03d}｜{len(members)}张｜'
                        f'{"⚠ 类别冲突" if conflict else "类别一致"}</h2><div class="gallery">{"".join(member_cards)}</div>'
                        f'<details><summary>查看簇内达标关系（{len(edges)}条）</summary><table><tr><th>图片A</th><th>图片B</th>'
                        f'<th>pHash相似度</th><th>SSIM</th></tr>{edge_html}</table></details></section>')
        if cluster_number % 25 == 0:
            print(f"Rendered {cluster_number}/{len(cross)} clusters", flush=True)

    fields = ["相似簇编号", "簇内序号", "图片", "数据划分", "二分类标签", "审核图", "人工结论", "是否保留", "备注"]
    with (output / "247个跨划分相似簇审核清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(manifest)
    style = '''body{font-family:"Microsoft YaHei",sans-serif;max-width:1800px;margin:auto;padding:20px;background:#f4f5f7;color:#202124}
section{background:white;padding:18px;margin:24px 0;border-radius:10px}.gallery{display:flex;gap:14px;overflow-x:auto;align-items:flex-start}
figure{margin:0;flex:0 0 auto;max-width:480px}img{max-height:700px;max-width:460px}figcaption{word-break:break-all;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccc;padding:5px}nav{position:sticky;top:0;background:#fff;padding:10px;z-index:5}'''
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>247个跨划分相似簇审核</title><style>{style}</style></head><body>'
    page += f'<nav><b>247个跨划分相似簇｜1095张带框图片</b>　<a href="#{conflict_id}">跳到类别冲突簇</a></nav>' + "".join(sections) + '</body></html>'
    (output / "index.html").write_text(page, encoding="utf-8")
    if conflict_id:
        shutil.copytree(output / "clusters" / conflict_id, output / "优先审核_3张类别冲突图")
        conflict_section = next(section for section in sections if f'id="{conflict_id}"' in section)
        (output / "优先审核_3张类别冲突图.html").write_text(
            f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{style}</style></head><body>{conflict_section}</body></html>', encoding="utf-8")
    (output / "说明.txt").write_text(
        "共247个跨train/val/test的视觉相似簇，涉及1095张图片。\n"
        "筛选条件：pHash相似度>=0.90且连接边SSIM>=0.85。簇由达标边传递连接，簇内并非任意两张都必须直接达标。\n"
        "所有图片均已绘制原YOLO框；本工具未修改或删除数据集文件。\n", encoding="utf-8")
    print(f"Done: clusters={len(cross)} images={len(manifest)} conflict={conflict_id} output={output}")


if __name__ == "__main__":
    main()
