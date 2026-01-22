#!/usr/bin/env python3
"""
Analyze extraction quality on a sample of PDFs.
"""
import argparse, json, time, random, sys
from pathlib import Path
from typing import List, Dict

# Re‑use the existing extractor logic (import from same folder)
try:
    # Adjust import path if script is moved elsewhere
    from extract_corpus import extract_pdf  # type: ignore
except Exception as e:
    print("Unable to import extract_pdf:", e, file=sys.stderr)
    sys.exit(1)


def gather_pdfs(src: Path, limit: int = 1000) -> List[Path]:
    """Return up to `limit` PDF paths (randomly shuffled for a representative sample)."""
    all_pdfs = list(src.rglob("*.pdf"))
    random.shuffle(all_pdfs)
    return all_pdfs[:limit]


def analyze_one(pdf_path: Path) -> Dict:
    info = {
        "path": str(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
        "pages": None,
        "text_len": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        start = time.time()
        import fitz
        doc = fitz.open(str(pdf_path))
        info["pages"] = len(doc)
        # Extract text page‑by‑page (same as extract_pdf)
        texts = [page.get_text("text") for page in doc]
        text = "\n".join(texts)
        elapsed = time.time() - start
        info.update({
            "text_len": len(text),
            "elapsed_s": round(elapsed, 3),
        })
    except Exception as exc:
        info["error"] = str(exc)
    return info


def main(src: Path, out: Path, limit: int):
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    pdfs = gather_pdfs(src, limit)
    results = [analyze_one(p) for p in pdfs]

    # Simple quality flags
    low_ratio = [r for r in results if not r["error"] and r["pages"] and r["text_len"] < 0.2 * r["size_bytes"]]
    empty_text = [r for r in results if not r["error"] and r["text_len"] == 0]
    slow_extraction = [
        r
        for r in results
        if not r["error"]
        and r["elapsed_s"]
        and (r["size_bytes"] / (1024 * 1024)) > 0
        and r["elapsed_s"] / (r["size_bytes"] / (1024 * 1024)) > 5.0
    ]

    summary = {
        "total_files": len(pdfs),
        "processed": sum(1 for r in results if not r["error"]),
        "failed": sum(1 for r in results if r["error"]),
        "average_seconds_per_mb": round(
            sum(r["elapsed_s"] for r in results if r.get("elapsed_s"))
        / max(sum(r["size_bytes"] for r in results) / (1024 * 1024), 1e-6)
    ),
        "low_text_ratio": [r["path"] for r in low_ratio[:10]],
        "empty_extractions": [r["path"] for r in empty_text[:10]],
        "slow_extractions": [r["path"] for r in slow_extraction[:10]],
    }

    out.write_text(json.dumps({"details": results, "summary": summary}, indent=2))
    print(f"Analysis written to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick quality analysis on PDF extraction")
    parser.add_argument("--src", type=Path, default=Path("/Volumes/8SSD/paper/pdfs"), help="Root folder containing PDFs")
    parser.add_argument("--out", type=Path, default=Path("extraction_analysis.json"), help="Where to store the JSON report")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of PDFs to analyse")
    args = parser.parse_args()
    main(args.src, args.out, args.limit)
