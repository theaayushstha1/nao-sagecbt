"""Render any Markdown doc in this repo to a print-quality PDF.

Unlike ``build_walkthrough_pdf.py`` (which hand-assembles a PDF with
reportlab), this converts an existing ``.md`` file, so the Markdown stays the
single source of truth and the PDF is a build artifact.

Usage:
    python docs/build_md_pdf.py docs/ENV_REFERENCE.md
    python docs/build_md_pdf.py docs/ENV_REFERENCE.md /tmp/out.pdf

Requires ``markdown``, ``pygments``, ``weasyprint`` and the native pango libs
(``conda install -c conda-forge pango``).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

# Strip the repo's HTML-comment frontmatter block (title:/tags:/related:).
_FRONTMATTER = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)

CSS = """
@page {
    size: Letter;
    margin: 0.85in 0.8in 0.9in 0.8in;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: -apple-system, "Helvetica Neue", sans-serif;
        font-size: 8.5pt; color: #8a94a6;
    }
    @top-right {
        content: string(doctitle);
        font-family: -apple-system, "Helvetica Neue", sans-serif;
        font-size: 8.5pt; color: #8a94a6;
    }
}
@page :first { @top-right { content: none; } }

body {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt; line-height: 1.55; color: #1c2430;
}

h1 {
    string-set: doctitle content();
    font-size: 23pt; color: #0B2545; margin: 0 0 .1in;
    padding-bottom: .07in; border-bottom: 3px solid #F25C05;
}
h2 {
    font-size: 14pt; color: #0B2545; margin: .3in 0 .08in;
    padding-bottom: .04in; border-bottom: 1px solid #dbe2ea;
    break-after: avoid; break-inside: avoid;
}
h3 { font-size: 11.5pt; color: #123a63; margin: .18in 0 .05in; break-after: avoid; }
p  { margin: .05in 0 .08in; orphans: 2; widows: 2; }

/* Inline code */
code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 8.8pt; background: #eef2f7; color: #0B2545;
    padding: .5pt 3pt; border-radius: 2px;
}
/* Fenced blocks */
pre {
    background: #f7f9fc; border: 1px solid #dbe2ea;
    border-left: 3px solid #F25C05; border-radius: 3px;
    padding: 7pt 9pt; margin: .08in 0;
    font-size: 8.5pt; line-height: 1.4;
    white-space: pre-wrap; word-wrap: break-word;
    break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }

table {
    border-collapse: collapse; width: 100%;
    margin: .1in 0; font-size: 8.8pt;
    break-inside: auto;
}
thead { display: table-header-group; }   /* repeat header across pages */
tr { break-inside: avoid; }
th {
    background: #0B2545; color: #fff; text-align: left;
    padding: 5pt 6pt; font-weight: 600;
}
td { padding: 4.5pt 6pt; border-bottom: 1px solid #e3e8ef; vertical-align: top; }
tbody tr:nth-child(even) { background: #f7f9fc; }
td code { font-size: 8pt; }

blockquote {
    margin: .1in 0; padding: 6pt 10pt;
    background: #fff8f2; border-left: 3px solid #F25C05;
    color: #4a3527; break-inside: avoid;
}
blockquote p { margin: .02in 0; }

ul, ol { margin: .05in 0 .1in; padding-left: .22in; }
li { margin: .025in 0; }

hr { border: none; border-top: 1px solid #dbe2ea; margin: .22in 0; }
a { color: #12507f; text-decoration: none; }
strong { color: #0B2545; }
"""


def build(md_path: Path, pdf_path: Path) -> Path:
    text = _FRONTMATTER.sub("", md_path.read_text(encoding="utf-8"))
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(pdf_path)
    return pdf_path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    md_path = Path(sys.argv[1]).resolve()
    if not md_path.is_file():
        print(f"error: no such file: {md_path}", file=sys.stderr)
        return 1
    pdf_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else md_path.with_suffix(".pdf")
    )
    build(md_path, pdf_path)
    print(f"wrote {pdf_path}  ({pdf_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
