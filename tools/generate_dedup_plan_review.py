"""Generate an HTML review page by joining rendered cluster images with a dry-run plan."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plans = list(csv.DictReader(args.plan.open(encoding="utf-8-sig")))
    rendered = {row["图片"]: row["审核图"] for row in csv.DictReader(args.render_manifest.open(encoding="utf-8-sig"))}
    groups = defaultdict(list)
    for row in plans:
        if row["图片"] not in rendered:
            raise ValueError(f"缺少审核图：{row['图片']}")
        groups[int(row["相似簇编号"])].append(row)
    sections = []
    for cluster_id in sorted(groups):
        rows = groups[cluster_id]
        cards = []
        for row in rows:
            dropped = row["处理决定"] == "拟删除"
            cls = "drop" if dropped else "keep"
            decision = "拟删除" if dropped else f"拟保留 → {row['拟数据划分']}"
            details = (f"{decision}<br>质量分 {row['质量分']}<br>"
                       f"对应代表帧：{html.escape(row['对应代表帧'])}")
            cards.append(f'<figure class="{cls}"><img loading="lazy" src="{html.escape(rendered[row["图片"]])}">'
                         f'<figcaption><b>{details}</b><br>{html.escape(row["图片"])}</figcaption></figure>')
        keep = sum(r["处理决定"] == "拟保留" for r in rows)
        drop = len(rows)-keep
        target = next((r["拟数据划分"] for r in rows if r["处理决定"] == "拟保留"), "")
        sections.append(f'<section><h2>相似簇 {cluster_id:03d}｜原{len(rows)}张｜拟保留{keep}｜拟删除{drop}｜拟统一进入{target}</h2>'
                        f'<div class="gallery">{"".join(cards)}</div></section>')
    css = '''body{font-family:"Microsoft YaHei",sans-serif;max-width:1800px;margin:auto;padding:20px;background:#f4f5f7}
nav{position:sticky;top:0;background:#fff;padding:12px;z-index:5}section{background:white;padding:16px;margin:22px 0;border-radius:10px}
.gallery{display:flex;gap:14px;overflow-x:auto;align-items:flex-start}figure{margin:0;padding:7px;flex:0 0 auto;max-width:470px;border:5px solid}
figure.keep{border-color:#2e7d32}figure.drop{border-color:#d32f2f;opacity:.75}img{max-height:700px;max-width:450px}figcaption{font-size:13px;word-break:break-all}
'''
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>v2抽帧预案审核</title><style>{css}</style></head><body>'
    page += '<nav><b>v2抽帧预案：绿色=拟保留，红色=拟删除。本页面仅供审核，尚未修改数据集。</b></nav>'
    page += "".join(sections) + '</body></html>'
    args.output.write_text(page, encoding="utf-8")
    print(f"Done: clusters={len(groups)} rows={len(plans)} output={args.output}")


if __name__ == "__main__":
    main()
