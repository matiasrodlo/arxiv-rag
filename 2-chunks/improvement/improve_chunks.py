#!/usr/bin/env python3
"""
metadata_augmentation.py

Create a compact JSON per paper that contains:
  • title, abstract, authors, arXiv ID / DOI (top‑level)
  • extracted keywords list
  • section_index, citations_map, figure placeholders, author contacts
  • summary and domain_tags (useful short‑text embeddings)
  • the original chunks array (for retrieval) plus a plain_text field
  • normalized citation IDs (e.g. "cite_12")
  • citation_spans: mapping each citation ID to its location in the source text

The script never overwrites the source files; it writes the new JSONs to a
separate directory (default: output_augmented/).  You may point it at a single
file or an entire folder.
"""

import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Helper extraction functions
# ----------------------------------------------------------------------

def extract_keywords(abstract: str) -> List[str]:
    match = re.search(r"Keywords?\s*[:：]\s*(.+)", abstract,
                      flags=re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1)
    parts = re.split(r"[;,]\s*", raw.strip())
    return [p.strip() for p in parts if p.strip()]

def build_section_index(chunks: List[dict]) -> Dict[str, Tuple[int, int]]:
    index: Dict[str, Tuple[int, int]] = {}
    for i, chunk in enumerate(chunks):
        name = chunk.get("name")
        if not name:
            continue
        if name not in index:
            index[name] = (i, i)
        else:
            start, _ = index[name]
            index[name] = (start, i)
    return index

def extract_citations(chunks: List[dict]) -> Dict[str, str]:
    """Return a map of raw citation number → full reference string."""
    citations: Dict[str, str] = {}
    cite_pat = re.compile(r"\[(\d+)\]\s*(.+)")
    for chunk in chunks:
        text = chunk.get("text", "")
        for line in text.splitlines():
            m = cite_pat.search(line)
            if m:
                cid, ref = m.groups()
                citations[cid] = ref.strip()
    return citations

def normalize_citation_ids(citations: Dict[str, str]) -> Dict[str, str]:
    """Convert numeric keys like "12" → "cite_12" for easier downstream replacement."""
    return {f"cite_{k}": v for k, v in citations.items()}

def extract_citation_spans(chunks: List[dict]) -> Dict[str, List[int]]:
    """Map each citation number to [chunk_idx, start_char, end_char].
    We search the raw text of every chunk; if a pattern ``[12]`` appears we record
    its location.  Only the *first* occurrence per chunk is stored – enough for UI
    highlighting.
    """
    spans: Dict[str, List[int]] = {}
    cite_pat = re.compile(r"\[(\d+)\]")
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        for m in cite_pat.finditer(text):
            cid = m.group(1)
            # store only the first span we encounter for a given citation
            if cid not in spans:
                spans[cid] = [idx, m.start(), m.end()]
    return spans

def extract_figure_placeholders(chunks: List[dict]) -> List[Dict[str, str]]:
    placeholders = []
    fig_pat = re.compile(r"(Figure|Fig\.? )\s*([0-9]+)", flags=re.IGNORECASE)
    for chunk in chunks:
        text = chunk.get("text", "")
        for line in text.splitlines():
            m = fig_pat.search(line)
            if m:
                fig_id = f"Fig {m.group(2)}"
                start = max(0, m.start() - 30)
                end = min(len(line), m.end() + 30)
                caption = line[start:end].strip()
                placeholders.append({"id": fig_id, "caption": caption})
    return placeholders

def extract_author_emails(chunks: List[dict]) -> List[str]:
    emails = set()
    email_pat = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    for chunk in chunks[:5]:  # header is usually at the top
        for match in email_pat.findall(chunk.get("text", "")):
            emails.add(match)
    return sorted(emails)

def generate_summary(abstract: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    return " ".join(sentences[:2])

def derive_domain_tags(keywords: List[str]) -> List[str]:
    tag_map = {
        "abstraction": "soft-constraints",
        "constraint solving": "constraint-satisfaction",
        "semiring homomorphism": "semiring-theory",
        "order–reflecting": "order-theory",
    }
    tags = {tag_map[kw.lower()] for kw in keywords if kw.lower() in tag_map}
    return sorted(tags)

def concatenate_plain_text(chunks: List[dict]) -> str:
    """Join all chunk texts into a single string (preserving order)."""
    return "\n\n".join(chunk.get("text", "") for chunk in chunks)

# ----------------------------------------------------------------------
# Build the light JSON payload (now includes DOI/arXiv, normalized IDs,
# citation spans, and plain_text)
# ----------------------------------------------------------------------

def build_light_payload(paper: dict) -> dict:
    abstract = paper.get("abstract", "")
    # Core metadata
    light = {
        "title": paper.get("title"),
        "abstract": abstract,
        "authors": paper.get("authors", []),
        "keywords_list": extract_keywords(abstract),
    }

    # Preserve a stable identifier – many papers already have an arXiv ID in the metadata.
    # We fall back to any DOI field that may exist; if none, we keep it empty.
    light["arxiv_id"] = paper.get("metadata", {}).get("paper_id") or ""
    light["doi"] = paper.get("metadata", {}).get("doi") or ""

    chunks = paper.get("chunks", [])
    if chunks:
        # Keep the full chunk list for retrieval
        light["chunks"] = chunks
        # Add a single concatenated plain_text field (useful for models with long context)
        light["plain_text"] = concatenate_plain_text(chunks)

        # Section index, figures, author contacts
        light.update({
            "section_index": build_section_index(chunks),
            "figures": extract_figure_placeholders(chunks),
            "author_contacts": extract_author_emails(chunks),
        })

        # Citations handling
        raw_cites = extract_citations(chunks)
        light["citations_map"] = normalize_citation_ids(raw_cites)
        light["citation_spans"] = extract_citation_spans(chunks)

    # Compact helpers for quick answering
    light["summary"] = generate_summary(abstract)
    light["domain_tags"] = derive_domain_tags(light.get("keywords_list", []))

    return light

# ----------------------------------------------------------------------
# CLI handling – process a file or an entire directory
# ----------------------------------------------------------------------

def process_path(input_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Walk ``input_path`` (file or directory), build the light JSON for each
    *.json file, and write it under ``output_dir`` preserving the relative path."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob("*.json"))

    for src in files:
        data = json.load(src.open(encoding="utf-8"))
        light = build_light_payload(data)

        # Destination keeps the same relative hierarchy
        rel = src.relative_to(input_path)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        with dst.open("w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False, indent=2)

        print(f"✅ Light JSON created → {dst}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact JSON per paper that contains the fields needed for "
            "embedding generation *and* extra citation‑friendly metadata."
        )
    )
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="File or directory with the original output_improved JSONs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("output_augmented"),
        help=(
            "Directory where the light JSON files will be written. The relative "
            "folder structure of ``input`` is preserved."
        ),
    )
    args = parser.parse_args()
    process_path(args.input, args.output_dir)

if __name__ == "__main__":
    main()
