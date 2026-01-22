#!/usr/bin/env python3
"""
Extract PDFs to a structured JSON corpus preserving hierarchy,
with optional OCR fallback and parallel processing.
"""
import argparse, json, sys, datetime, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Optional dependencies – fail early with a clear message
# ----------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF required: pip install pymupdf")

# OCR is optional; if not present we simply skip OCR fallback.
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --------------------------------------------------------------
# Helpers for section detection
# --------------------------------------------------------------
def _get_page_blocks(page) -> list:
    """Return a list of text blocks with their average font size.
    Each block dict contains:
        - text : concatenated string of the block
        - size : average font size (float)
    """
    page_dict = page.get_text("dict")
    blocks = []
    for b in page_dict["blocks"]:
        if b.get("type") != 0:  # skip non‑text blocks
            continue
        txt_parts = []
        sizes = []
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

def _build_hierarchy(blocks: list):
    """Create a simple heading hierarchy from page blocks.
    Heuristic:
      * Largest font size on the page → level‑1 heading.
      * Anything >= 0.8 × max_size → sub‑heading (level‑2).
      * Remaining blocks are body text attached to the most recent heading.
    Returns a list of sections: {"title", "level", "content"}.
    """
    if not blocks:
        return []
    max_size = max(b["size"] for b in blocks)
    sections = []
    current = None
    for blk in blocks:
        txt, size = blk["text"], blk["size"]
        # treat as heading when font is relatively large
        if size >= 0.8 * max_size:
            if current:
                sections.append(current)
            level = 1 if size == max_size else 2
            current = {"title": txt, "level": level, "content": ""}
        else:
            if not current:
                # no heading yet – create a dummy top‑level section
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

def extract_pdf(pdf_path: Path):
    """Extract a PDF into a list of pages with section hierarchy.
    Each page entry looks like:
        {"page": <int>, "sections": [{"title", "level", "content"}, ...]}
    OCR is applied to pages that contain no extractable text.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        # Try to get structured blocks first
        blocks = _get_page_blocks(page)
        if not blocks:
            # Possibly a scanned image‑only page – fall back to OCR as one block
            ocr_text = _ocr_page(page) if OCR_AVAILABLE else ""
            sections = [{"title": "", "level": 1, "content": ocr_text}]
        else:
            sections = _build_hierarchy(blocks)
        pages.append({"page": i, "sections": sections})
    return pages

def _process_file(pdf_path: Path, src_dir: Path, dst_dir: Path) -> None:
    rel_path = pdf_path.relative_to(src_dir).with_suffix('.json')
    out_path = dst_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pages = extract_pdf(pdf_path)
        payload = {
            "metadata": {
                "source": str(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "num_pages": len(pages),
                "extracted_at": datetime.datetime.utcnow().isoformat() + "Z",
            },
            "pages": pages,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Extracted: {pdf_path} → {out_path}")
    except Exception as exc:
        print(f"Failed to extract {pdf_path}: {exc}", file=sys.stderr)

def main(src_dir: Path, dst_dir: Path) -> None:
    src_dir = src_dir.expanduser().resolve()
    dst_dir = dst_dir.expanduser().resolve()
    if not src_dir.is_dir():
        sys.exit(f"Source directory does not exist: {src_dir}")
    pdf_paths = list(src_dir.rglob('*.pdf'))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_file, p, src_dir, dst_dir): p for p in pdf_paths}
        for _ in as_completed(futures):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF corpus with sections & OCR")
    parser.add_argument("--src", type=Path, default=Path("/Volumes/8SSD/paper/pdfs"), help="Root folder containing PDFs")
    parser.add_argument("--dst", type=Path, default=Path("corpus_json"), help="Destination folder for extracted JSON files")
    args = parser.parse_args()
    main(args.src, args.dst)
