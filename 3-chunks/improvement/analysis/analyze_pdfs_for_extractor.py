#!/usr/bin/env python3
"""
Analyze a large collection of PDFs to surface structural patterns that can guide improvements
for the :pyfunc:`extract_corpus.extract_pdf` extractor.

The script walks through a directory (default: ``/Volumes/8SSD/paper/pdfs``), samples up to
1000 PDF files, extracts each with the existing ``extract_corpus`` logic and gathers
statistics such as:

* total pages per document
* number of heading levels detected (level‑1 vs level‑2)
* proportion of pages that required OCR fallback
* average block font size distribution
* presence of very long sections (> 2000 characters) which may need chunking

The aggregated results are written to ``analysis_report.json`` (a list of per‑file stats)
and a concise CSV ``summary.csv`` for quick inspection.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Optional dependencies – the same as in extract_corpus.py
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit("PyMuPDF required: pip install pymupdf")

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:  # pragma: no cover
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Re‑use the core helpers from extract_corpus (copied locally to avoid a hard import)
# ---------------------------------------------------------------------------
def _get_page_blocks(page) -> List[Dict]:
    """Return text blocks with their average font size.

    The implementation mirrors ``extract_corpus._get_page_blocks`` so that the
    analysis uses exactly the same heuristics as the extractor.
    """
    page_dict = page.get_text("dict")
    blocks: List[Dict] = []
    for b in page_dict["blocks"]:
        if b.get("type") != 0:
            continue
        txt_parts, sizes = [], []
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if txt:
                    txt_parts.append(txt)
                    sizes.append(span.get("size", 0))
        if not txt_parts:
            continue
        block_text = " ".join(txt_parts).strip()
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        blocks.append({"text": block_text, "size": avg_size})
    return blocks


def _build_hierarchy(blocks: List[Dict]):
    """Create a simple heading hierarchy from page blocks.

    Returns a list of sections ``{"title", "level", "content"}``.
    """
    if not blocks:
        return []
    max_size = max(b["size"] for b in blocks)
    sections: List[Dict] = []
    current = None
    for blk in blocks:
        txt, size = blk["text"], blk["size"]
        if size >= 0.8 * max_size:
            if current:
                sections.append(current)
            level = 1 if size == max_size else 2
            current = {"title": txt, "level": level, "content": ""}
        else:
            if not current:
                current = {"title": "", "level": 1, "content": txt + "\n"}
            else:
                current["content"] += txt + "\n"
    if current:
        sections.append(current)
    return sections


def _ocr_page(page) -> str:
    """Run Tesseract OCR on a page (fallback for empty native text)."""
    if not OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(image)


def extract_pdf(pdf_path: Path) -> List[Dict]:
    """Extract a PDF using the same logic as ``extract_corpus.extract_pdf``.

    Returns a list of page dictionaries with ``page`` number and ``sections``.
    """
    doc = fitz.open(str(pdf_path))
    pages: List[Dict] = []
    for i, page in enumerate(doc, start=1):
        blocks = _get_page_blocks(page)
        if not blocks:
            ocr_text = _ocr_page(page) if OCR_AVAILABLE else ""
            sections = [{"title": "", "level": 1, "content": ocr_text}]
        else:
            sections = _build_hierarchy(blocks)
        pages.append({"page": i, "sections": sections})
    return pages

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def analyse_document(pdf_path: Path) -> Dict:
    """Run ``extract_pdf`` on *pdf_path* and compute a set of metrics.

    Returns a dictionary that can be JSON‑serialised.
    """
    try:
        pages = extract_pdf(pdf_path)
    except Exception as exc:  # pragma: no cover – defensive, should rarely happen
        return {"path": str(pdf_path), "error": str(exc)}

    total_pages = len(pages)
    ocr_pages = 0
    heading_counts = Counter()
    block_sizes: List[float] = []
    long_section_count = 0

    for p in pages:
        sections = p["sections"]
        # Detect if this page fell back to OCR (no blocks → single empty‑title section)
        if len(sections) == 1 and not sections[0]["title"] and not sections[0]["content"].strip():
            # No text at all – treat as OCR failure, not a success
            pass
        elif any(s["content"] == "" for s in sections):
            # If we used the OCR branch, content will be populated but title empty
            ocr_pages += 1
        for sec in sections:
            heading_counts[sec["level"]] += 1 if sec["title"] else 0
            block_sizes.append(len(sec["content"]))
            if len(sec["content"]) > 2000:
                long_section_count += 1

    avg_block_size = sum(block_sizes) / len(block_sizes) if block_sizes else 0

    return {
        "path": str(pdf_path),
        "num_pages": total_pages,
        "ocr_page_ratio": ocr_pages / total_pages if total_pages else 0,
        "heading_counts": dict(heading_counts),
        "avg_section_length": avg_block_size,
        "long_sections_gt_2000_chars": long_section_count,
    }


def main(src_dir: Path, max_files: int = 1000):
    src_dir = src_dir.expanduser().resolve()
    if not src_dir.is_dir():
        sys.exit(f"Source directory does not exist: {src_dir}")

    pdf_paths = list(src_dir.rglob("*.pdf"))[:max_files]
    print(f"Analyzing {len(pdf_paths)} PDF files …")

    results: List[Dict] = []
    for p in pdf_paths:
        results.append(analyse_document(p))

    # Write detailed JSON report
    out_json = src_dir.parent / "analysis_report.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed report written to {out_json}")

    # Produce a CSV summary for quick viewing
    import csv
    out_csv = src_dir.parent / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "path",
            "num_pages",
            "ocr_page_ratio",
            "h1_count",
            "h2_count",
            "avg_section_length",
            "long_sections_gt_2000_chars",
        ]
        writer.writerow(header)
        for r in results:
            if "error" in r:
                continue
            writer.writerow([
                r["path"],
                r["num_pages"],
                f"{r['ocr_page_ratio']:.2%}",
                r["heading_counts"].get(1, 0),
                r["heading_counts"].get(2, 0),
                f"{r['avg_section_length']:.1f}",
                r["long_sections_gt_2000_chars"],
            ])
    print(f"CSV summary written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PDFs to guide improvements for extract_corpus")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/Volumes/8SSD/paper/pdfs"),
        help="Root folder containing PDFs to analyse",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of PDF files to process (default 1000)",
    )
    args = parser.parse_args()
    main(args.src, args.max_files)
