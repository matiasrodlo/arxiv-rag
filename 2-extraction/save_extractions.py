#!/usr/bin/env python3
"""
Utility script to extract full text from all PDFs in the default input directory
and save each paper as a .txt file.

The extraction uses the high‑quality PDFExtractor (OCR disabled for speed).
Output files are written to /Volumes/8SSD/paper/extracted_texts/<paper_id>.txt
"""
import sys
from pathlib import Path
import importlib.util

# Add repository root to sys.path so relative imports work
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

# Dynamically load the PDFExtractor module (avoid package name issues)
pdf_extractor_path = repo_root / "2-extraction" / "pdf_extractor.py"
spec = importlib.util.spec_from_file_location("pdf_extractor", pdf_extractor_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PDFExtractor = mod.PDFExtractor
DEFAULT_PDF_INPUT_DIR = mod.DEFAULT_PDF_INPUT_DIR


def main():
    input_dir: Path = DEFAULT_PDF_INPUT_DIR
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path("/Volumes/8SSD/paper/extracted_texts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclude hidden macOS resource‑fork files (those starting with '._')
    pdf_files = [p for p in input_dir.rglob("*.pdf") if not p.name.startswith('._')][:100]
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")

    # Fast extractor (OCR disabled) for the first pass
    # ----------------------------------------------------------------------
    # Parallel extraction – keep the system stable based on available CPUs
    # ----------------------------------------------------------------------
    import os, concurrent.futures, multiprocessing as mp

    fast_extractor = PDFExtractor(enable_ocr=False)
    OCR_QUALITY_THRESHOLD = 0.85   # Run OCR only when quality_score < 0.85
    max_workers = max(1, min(4, (os.cpu_count() or 2) - 1))  # leave one core free

    def process_one(idx, pdf_path):
        try:
            # ---- First extraction (fast) ----
            result = fast_extractor.extract(str(pdf_path))
            if not result.get("success"):
                return f"{idx}/{len(pdf_files)} [WARN] Extraction reported failure for {pdf_path.name}", None
            # ---- Check quality, possibly redo with OCR ----
            q = result.get('quality_score', 0.0)
            if q < OCR_QUALITY_THRESHOLD:
                ocr_extractor = PDFExtractor(enable_ocr=True)
                ocr_result = ocr_extractor.extract(str(pdf_path))
                if ocr_result.get('quality_score', 0.0) > q:
                    result = ocr_result
            # ---- Clean text ----
            def clean_extra_headers(text: str) -> str:
                import re
                text = re.sub(r'^\s*\d+\s*$\n?', '', text, flags=re.MULTILINE)
                text = re.sub(r'^(Proceedings|Conference|Workshop|©).*?\n', '', text,
                              flags=re.IGNORECASE | re.MULTILINE)
                return text
            result["text"] = clean_extra_headers(result["text"]) 
            import unicodedata
            result["text"] = unicodedata.normalize('NFKC', result["text"]) 
            # ---- Table extraction (pdfplumber) ----
            tables = None
            try:
                import pdfplumber
                with pdfplumber.open(str(pdf_path)) as pdf:
                    all_tables = []
                    for page in pdf.pages:
                        page_tables = page.extract_tables()
                        if page_tables:
                            all_tables.extend(page_tables)
                    if all_tables:
                        tables = all_tables
            except Exception:
                pass
            # ---- Build JSON data and write file ----
            rel_path = pdf_path.relative_to(DEFAULT_PDF_INPUT_DIR).with_suffix('')
            json_path = output_dir / rel_path.with_suffix('.json')
            json_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            # ---- Normalize metadata fields (trim whitespace, title‑case) ----
            def _clean_str(s):
                return s.strip() if isinstance(s, str) else s
            meta = result.get("metadata", {}) or {}
            normalized_meta = {
                "title": _clean_str(meta.get("title", "")),
                "author": _clean_str(meta.get("author", "")),
                "subject": _clean_str(meta.get("subject", "")),
                "creator": _clean_str(meta.get("creator", "")),
                "producer": _clean_str(meta.get("producer", "")),
                "creation_date": _clean_str(meta.get("creation_date", "")),
                "modification_date": _clean_str(meta.get("modification_date", "")),
                "page_count": meta.get("page_count")
            }
            # Title‑case for nicer display
            if normalized_meta["title"]:
                normalized_meta["title"] = normalized_meta["title"].title()
            if normalized_meta["author"]:
                normalized_meta["author"] = normalized_meta["author"].title()

            data = {
                "text": result["text"],
                "metadata": normalized_meta,
                "pages": result.get("pages", []),
                "method_used": result.get("method_used"),
                "quality_score": result.get("quality_score"),
                "tables": tables,
            }
            json_path.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            msg = f"{idx}/{len(pdf_files)} OK  saved {json_path} ({result.get('metadata',{}).get('page_count','?')} pages, score={result.get('quality_score'):.3f})"
            return msg, None
        except Exception as e:
            return f"{idx}/{len(pdf_files)} [ERROR] {pdf_path.name}: {e}", None

    # Run in a process pool (better for CPU‑bound OCR)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, i+1, p): (i+1, p) for i, p in enumerate(pdf_files)}
        for fut in concurrent.futures.as_completed(futures):
            msg, _ = fut.result()
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
