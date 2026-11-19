"""HTML and Markdown report export."""
from __future__ import annotations

import datetime as dt
import html

import pandas as pd

ICON = {"warn": "⚠️", "good": "✅", "info": "ℹ️"}


def _md_inline(text: str) -> str:
    """Tiny markdown -> HTML for **bold** and `code` in findings."""
    import re
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def build_html(title: str, dataset: str, quality: dict, findings: list[dict], tables: dict[str, pd.DataFrame],
               figures: list, ml_section: dict | None = None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#1d2733}}
h1{{color:#2E86AB}} h2{{border-bottom:2px solid #2E86AB;padding-bottom:4px;margin-top:36px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%}} td,th{{border:1px solid #dde3ea;padding:4px 8px;text-align:left}}
th{{background:#f1f5f9}} .f{{padding:6px 10px;margin:4px 0;border-radius:6px}} .warn{{background:#fff4e5}} .good{{background:#e8f7ee}} .info{{background:#eef4fb}}
.grade{{font-size:42px;font-weight:700;color:#2E86AB}} code{{background:#f1f5f9;padding:1px 4px;border-radius:3px}}</style></head><body>
<h1>{html.escape(title)}</h1><p>Dataset: <b>{html.escape(dataset)}</b> &middot; generated {now} by Auto EDA + ML Studio</p>
<h2>Data quality</h2><p><span class="grade">{quality['grade']}</span> &nbsp; score {quality['score']}/100</p>
{pd.DataFrame([quality['components']]).to_html(index=False)}
<h2>Key findings</h2>"""]
    for f in findings:
        parts.append(f'<div class="f {f["level"]}">{ICON[f["level"]]} {_md_inline(f["text"])}</div>')
    if ml_section:
        parts.append("<h2>Machine learning</h2>")
        for k, v in ml_section.items():
            parts.append(f"<h3>{html.escape(k)}</h3>")
            parts.append(v.to_html(index=False) if isinstance(v, pd.DataFrame) else f"<p>{_md_inline(str(v))}</p>")
    for name, t in tables.items():
        if t is not None and not t.empty:
            parts.append(f"<h2>{html.escape(name)}</h2>{t.round(4).to_html()}")
    if figures:
        parts.append("<h2>Charts</h2>")
        for fig in figures:
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</body></html>")
    return "\n".join(parts)


def build_markdown(title: str, dataset: str, quality: dict, findings: list[dict], tables: dict[str, pd.DataFrame],
                   ml_section: dict | None = None) -> str:
    lines = [f"# {title}", "", f"Dataset: **{dataset}** - generated {dt.datetime.now():%Y-%m-%d %H:%M}", "",
             "## Data quality", "", f"**Grade {quality['grade']}** (score {quality['score']}/100)", ""]
    lines += [f"- {k}: {v}" for k, v in quality["components"].items()]
    lines += ["", "## Key findings", ""] + [f"- {ICON[f['level']]} {f['text']}" for f in findings]
    if ml_section:
        lines += ["", "## Machine learning", ""]
        for k, v in ml_section.items():
            lines += [f"### {k}", "", v.to_markdown(index=False) if isinstance(v, pd.DataFrame) else str(v), ""]
    for name, t in tables.items():
        if t is not None and not t.empty:
            lines += ["", f"## {name}", "", t.round(4).to_markdown()]
    return "\n".join(lines)
