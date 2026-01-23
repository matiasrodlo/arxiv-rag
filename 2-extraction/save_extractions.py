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


def get_quality_threshold(page_count: int) -> float:
    """Dynamic OCR quality threshold based on PDF size.

    Short papers tolerate lower scores, long papers demand higher confidence.
    """
    if page_count <= 5:
        return 0.70
    if page_count > 30:
        return 0.90
    return 0.85

def main():
    input_dir: Path = DEFAULT_PDF_INPUT_DIR
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path("/Volumes/8SSD/paper/extracted_texts")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Open a simple JSON‑lines log for monitoring and an error log
    import json, datetime
    log_path = output_dir / "extraction_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")
    error_log_path = output_dir / "extraction_errors.jsonl"
    error_file = open(error_log_path, "a", encoding="utf-8")
    # Exclude hidden macOS resource‑fork files (those starting with '._')
    pdf_files = [p for p in input_dir.rglob("*.pdf") if not p.name.startswith('._')]
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")

    # Process in chunks to limit memory usage (e.g., 5 000 PDFs per chunk)
    CHUNK_SIZE = 5000

    # Fast extractor (OCR disabled) for the first pass
    # ----------------------------------------------------------------------
    # Parallel extraction – keep the system stable based on available CPUs
    # ----------------------------------------------------------------------
    import os, concurrent.futures, multiprocessing as mp

    fast_extractor = PDFExtractor(enable_ocr=False)
    # Dynamic threshold function will be used per PDF
    # Placeholder; actual threshold computed later based on page count
    cpu_cnt = os.cpu_count() or 2
    max_workers = max(1, cpu_cnt - 1)  # leave one core free

    def process_one(idx, pdf_path):
        try:
            # ---- Cache check & first extraction (fast) ----
            cache_key = fast_extractor._get_cache_key(pdf_path)
            cached_meta = fast_extractor._load_from_cache(cache_key)
            if cached_meta:
                # Use cached page count (if present) to compute a dynamic OCR threshold
                cached_pages = cached_meta.get('page_count')
                if not cached_pages:
                    # Fallback: estimate from file size (approx 1 page per 50 KB)
                    try:
                        size_kb = pdf_path.stat().st_size / 1024
                        cached_pages = max(1, int(size_kb / 50))
                    except Exception:
                        cached_pages = 5
                dyn_thresh_cached = get_quality_threshold(cached_pages)
                if cached_meta.get('quality_score', 0) >= dyn_thresh_cached:
                    # Cached high‑quality result exists – reuse metadata, but still need full text extraction for the actual content
                    result = fast_extractor.extract(str(pdf_path))
                else:
                    # Cached entry below threshold – run fast extraction anyway (OCR may be triggered later)
                    result = fast_extractor.extract(str(pdf_path))
            else:
                # No cache – run fast extraction
                result = fast_extractor.extract(str(pdf_path))
            if not result.get("success"):
                return f"{idx}/{len(pdf_files)} [WARN] Extraction reported failure for {pdf_path.name}", None
            # ---- Check quality, possibly redo with OCR ----
            q = result.get('quality_score', 0.0)
            # Determine dynamic threshold based on page count (fallback to 5 pages if unknown)
            page_cnt_est = len(result.get('pages', [])) or 5
            dyn_thresh = get_quality_threshold(page_cnt_est)
            if q < dyn_thresh:
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
            # ---- Table extraction (pdfplumber) with filtering ----
            tables = None
            try:
                import pdfplumber
                with pdfplumber.open(str(pdf_path)) as pdf:
                    all_tables = []
                    for page in pdf.pages:
                        page_tables = page.extract_tables()
                        if page_tables:
                            all_tables.extend(page_tables)
                    # Filter out tiny tables (less than 2 rows or <3 columns) and deduplicate
                    filtered = []
                    seen_hashes = set()
                    for tbl in all_tables:
                        if not tbl:
                            continue
                        row_count = len(tbl)
                        col_count = max((len(r) for r in tbl), default=0)
                        if row_count < 2 or col_count < 3:
                            continue
                        # Simple deduplication by hash of stringified rows
                        tbl_hash = hash(str(tbl))
                        if tbl_hash in seen_hashes:
                            continue
                        seen_hashes.add(tbl_hash)
                        filtered.append(tbl)
                    if filtered:
                        tables = filtered
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
            # Write monitoring entry
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + 'Z',
                "pdf_path": str(pdf_path),
                "output_json": str(json_path),
                "page_count": result.get('metadata',{}).get('page_count'),
                "quality_score": result.get('quality_score'),
                "method_used": result.get('method_used'),
                "ocr_used": result.get('pdf_type') == 'scanned' or (result.get('method_used') == 'ocr'),
                "tables_extracted": bool(tables),
                "status": "success"
            }
            log_file.write(json.dumps(log_entry) + '\n')
            return msg, None
        except Exception as e:
            return f"{idx}/{len(pdf_files)} [ERROR] {pdf_path.name}: {e}", None

    # Run in chunks with a thread pool (functions need to be picklable)
    from tqdm import tqdm
    for start in range(0, len(pdf_files), CHUNK_SIZE):
        chunk = pdf_files[start:start + CHUNK_SIZE]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, i+1+start, p): (i+1+start, p)
                       for i, p in enumerate(chunk)}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(chunk),
                            desc=f"Processing PDFs {start + 1}-{min(start + CHUNK_SIZE, len(pdf_files))}"):
                try:
                    msg, _ = fut.result()
                    if msg:
                        print(msg)
                except Exception as e:
                    # Log unexpected future exception
                    err_entry = {
                        "timestamp": datetime.datetime.utcnow().isoformat() + 'Z',
                        "error": str(e)
                    }
                    error_file.write(json.dumps(err_entry) + '\n')
    # Close log files
    log_file.close()
    error_file.close()


if __name__ == "__main__":
    main()
