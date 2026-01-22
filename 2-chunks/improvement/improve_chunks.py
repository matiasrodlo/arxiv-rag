import argparse
import json
import pathlib
import re
import logging
import sys
from collections import Counter
from typing import Dict, List, Tuple

# Simple token estimate (≈0.75 words per token)
def estimate_tokens(word_count: int) -> int:
    return max(1, int(word_count / 0.75))

# Flesch Reading Ease calculation (basic approximation)
def flesch_reading_ease(text: str) -> float:
    if not text:
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)
    # Approximate syllable count by counting vowel groups per word
    syllable_count = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
    if not sentences or not words:
        return 0.0
    asl = len(words) / len(sentences)  # average sentence length (words)
    asw = syllable_count / len(words)   # average syllables per word
    return 206.835 - 1.015 * asl - 84.6 * asw
# Configure a simple file logger for warnings and errors
logging.basicConfig(
    filename="metadata_augmentation.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
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
    """Return a map of raw citation identifiers → full reference string.
    Supports a broad set of citation styles:
      • Numeric brackets ``[12]``
      • Author‑year ``(Doe, 2020)``
      • Plain superscript numbers ``12``
      • LaTeX ``\\cite{key}``
      • DOI strings ``doi:10.xxxx/...``
      • Author‑et al. patterns like ``Smith et al., (2021)``
    """
    citations: Dict[str, str] = {}
    # Expanded regex list for diverse citation formats
    patterns = [
        re.compile(r"\[(\d+)\]\s*(.+)"),                     # [12] Reference
        re.compile(r"\(([^)]+?,\s*\d{4}[a-z]?)\)"),          # (Doe, 2020)
        re.compile(r"(?<!\w)(\d{1,3})\s*(?=–|—|,|\.)"),       # plain numeric superscript
        re.compile(r"\\cite\{([^}]+)\}"),                    # LaTeX \cite{key}
        re.compile(r"doi:\s*10\.\d+/\S+", re.IGNORECASE),   # DOI strings
        re.compile(r"[A-Z][a-z]+ et al\.?[,]?\s*\(\d{4}\)")  # Author‑et al. (Year)
    ]
    for chunk in chunks:
        text = chunk.get("text", "")
        for line in text.splitlines():
            for pat in patterns:
                m = pat.search(line)
                if m:
                    # Determine citation id and reference text
                    if pat.pattern.startswith(r"\\cite"):
                        cid = m.group(1)
                        ref = line.strip()
                    elif pat.pattern.startswith(r"doi"):
                        cid = "doi"
                        ref = m.group(0)
                    else:
                        cid = m.group(1)
                        ref = line.strip()
                    citations[cid] = ref.strip()
    return citations

def normalize_citation_ids(citations: Dict[str, str]) -> Dict[str, str]:
    """Convert numeric keys like \"12\" → \"cite_12\" for easier downstream replacement."""
    return {f"cite_{k}": v for k, v in citations.items()}

def extract_citation_spans(chunks: List[dict]) -> Dict[str, List[int]]:
    """Map each citation identifier to its first occurrence location.
    Supports the same patterns as ``extract_citations``.
    Returns [chunk_idx, start_char, end_char] for the match.
    """
    spans: Dict[str, List[int]] = {}
    patterns = [
        re.compile(r"\[(\d+)\]"),                     # [12]
        re.compile(r"\(([^)]+?,\s*\d{4}[a-z]?)\)"), # (Doe, 2020)
        re.compile(r"(?<!\w)(\d{1,3})\s*(?=–|—|,|\.)")# plain superscript
    ]
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        for pat in patterns:
            for m in pat.finditer(text):
                cid = m.group(1)
                if cid not in spans:
                    spans[cid] = [idx, m.start(), m.end()]
    return spans

def extract_figure_placeholders(chunks: List[dict]) -> List[Dict[str, str]]:
    """Extract figure identifiers and their captions (if present)."""
    placeholders = []
    fig_pat = re.compile(r"(Figure|Fig\.? )\s*([0-9]+)([:.\-]?\s*)(.*)", flags=re.IGNORECASE)
    for chunk in chunks:
        text = chunk.get("text", "")
        for line in text.splitlines():
            m = fig_pat.search(line)
            if m:
                fig_id = f"Fig {m.group(2)}"
                caption = m.group(5).strip() or line.strip()
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

def build_light_payload(paper: dict, *, min_readability: float = 0.0, min_citations: int = 0) -> dict:
    abstract = paper.get("abstract", "").strip()
    # After we compute readability later, we'll filter based on thresholds
    if not abstract and paper.get("chunks"):
        first_text = paper["chunks"][0].get("text", "")
        abstract = ". ".join(first_text.split('. ')[:2]) + '.' if first_text else ""

    title = paper.get("title") or f"arXiv:{paper.get('metadata', {}).get('paper_id', '')}"

    light = {
        "title": title,
        "abstract": abstract,
        "authors": paper.get("authors", []),
        "keywords_list": extract_keywords(abstract),
    }

    # stable identifiers
    light["arxiv_id"] = paper.get("metadata", {}).get("paper_id") or ""
    light["doi"] = paper.get("metadata", {}).get("doi") or ""

    chunks = paper.get("chunks", [])
    if chunks:
        # Build plain text from main sections and also include bibliography/references if present
        plain_text = concatenate_plain_text(chunks)
        # Append references section (if stored under a known key) to improve citation detection
        if isinstance(paper, dict):
            for ref_key in ("references", "bibliography", "refs"):
                refs = paper.get(ref_key)
                if isinstance(refs, list):
                    plain_text += "\n\n" + " ".join(str(r) for r in refs)
        light["chunks"] = chunks
        light["plain_text"] = plain_text
        # Token budget enforcement (example 8000 tokens)
        MAX_TOKENS = 8000
        token_est = estimate_tokens(len(plain_text.split()))
        if token_est > MAX_TOKENS:
            logging.warning(f"File {paper.get('metadata', {}).get('paper_id')} exceeds token budget ({token_est} > {MAX_TOKENS}); truncating.")
            allowed_words = int(MAX_TOKENS * 0.75)
            plain_text = " ".join(plain_text.split()[:allowed_words])
            light["plain_text"] = plain_text
        # Keyword fallback using distinct words (TF‑IDF style)
        if not light["keywords_list"]:
            words = re.findall(r"\b\w{4,}\b", plain_text.lower())
            freq = Counter(words)
            threshold = len(words) * 0.05
            distinct = [w for w, c in freq.items() if c < threshold]
            light["keywords_list"] = sorted(distinct, key=freq.get, reverse=True)[:10]
        # Readability score
        light["readability_flesch"] = flesch_reading_ease(plain_text)
        # Token estimate for downstream use
        light["token_estimate"] = token_est

        light.update({
            "section_index": build_section_index(chunks),
            "figures": extract_figure_placeholders(chunks),
            "author_contacts": extract_author_emails(chunks),
        })

        raw_cites = extract_citations(chunks)
        # Apply citation count filter
        citation_count = len(raw_cites)
        if citation_count < min_citations:
            logging.info(f"Skipping {paper.get('metadata', {}).get('paper_id')} due to low citation count ({citation_count} < {min_citations})")
            return None
        cleaned_map = {}
        for cid, ref in raw_cites.items():
            clean_ref = ref.strip()
            while clean_ref and clean_ref[0] in ",.; ":
                clean_ref = clean_ref[1:]
            while clean_ref and clean_ref[-1] in ",.; ":
                clean_ref = clean_ref[:-1]
            cleaned_map[f"cite_{cid}"] = clean_ref
        light["citations_map"] = cleaned_map
        light["citation_spans"] = extract_citation_spans(chunks)

    light["summary"] = generate_summary(abstract)
    light["domain_tags"] = derive_domain_tags(light.get("keywords_list", []))
    return light

# ----------------------------------------------------------------------
# CLI handling – process a file or an entire directory (default: output/ -> output_augmented/)
# ----------------------------------------------------------------------

def process_path(input_path: pathlib.Path, output_root: pathlib.Path, *, min_readability: float = 0.0, min_citations: int = 0):
    if input_path.is_dir():
        for child in input_path.rglob("*.json"):
            rel = child.relative_to(input_path)
            out_file = output_root / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                paper = json.loads(child.read_text())
                payload = build_light_payload(paper, min_readability=min_readability, min_citations=min_citations)
                if payload is None:
                    # Skip low‑quality document
                    continue
                out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            except Exception as e:
                logging.error(f"Failed processing {child}: {e}")
    else:
        # single file
        out_file = output_root / input_path.name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            paper = json.loads(input_path.read_text())
            payload = build_light_payload(paper, min_readability=min_readability, min_citations=min_citations)
            if payload is None:
                return
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as e:
            logging.error(f"Failed processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create light JSON payloads from raw paper JSONs.")
    parser.add_argument("--input-dir", type=str, default="output",
                        help="Directory containing the original JSON files (default: output)")
    parser.add_argument("--output-dir", type=str, default="output_augmented",
                        help="Where to write the augmented JSONs (default: output_augmented)")
    # New quality‑filter thresholds
    parser.add_argument("--min-readability", type=float, default=0.0,
                        help="Minimum Flesch reading ease score to keep a document")
    parser.add_argument("--min-citations", type=int, default=0,
                        help="Minimum number of citations required to keep a document")
    args = parser.parse_args()

    input_root = pathlib.Path(args.input_dir).resolve()
    output_root = pathlib.Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        logging.error(f"Input directory {input_root} does not exist")
        return

    process_path(input_root, output_root,
                 min_readability=args.min_readability,
                 min_citations=args.min_citations)

if __name__ == "__main__":
    main()
