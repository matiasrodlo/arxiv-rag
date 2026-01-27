"""
PDF Text Extraction Module with Multi-Library Fallback
Handles robust extraction from various PDF formats including scanned documents.
Default PDF input directory is set to /Volumes/8SSD/paper/pdfs.
"""

import os
import re
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf not available")

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available")


DEFAULT_PDF_INPUT_DIR = Path(
    os.environ.get('ARXIV_RAG_PDF_DIR', '/Volumes/8SSD/paper/pdfs')
)

# Cache directory: supports environment variable override
DEFAULT_CACHE_DIR = Path(
    os.environ.get('ARXIV_RAG_CACHE_DIR', str(Path.home() / '.arxiv_rag_cache'))
)

# Pre-compiled regex patterns for post-processing (compiled once at module load)
# Using tuples of (pattern, replacement) for simple replacements
_POST_PATTERNS_SIMPLE = (
    # 1. Fix broken sentences (period followed by lowercase) - improved pattern
    (re.compile(r'\.\s+([a-z])'), r'. \1'),
    # 2. Fix broken words across lines (hyphen at end of line)
    (re.compile(r'-\s*\n\s*'), ''),
    # 3. Fix broken URLs and emails - improved pattern
    (re.compile(r'([a-zA-Z0-9])\s+([@.])\s+([a-zA-Z0-9])'), r'\1\2\3'),
    # 4. Fix broken mathematical expressions - spacing around operators
    (re.compile(r'([a-zA-Z0-9])\s*([+\-*/=<>])\s*([a-zA-Z0-9])'), r'\1 \2 \3'),
    # 5. Fix broken citations [1] -> [1]
    (re.compile(r'\[\s*(\d+)\s*\]'), r'[\1]'),
    # 6. Fix broken references - improved pattern
    (re.compile(r'(Figure|Table|Equation|Section|Algorithm)\s+\n\s*(\d+)', re.IGNORECASE), r'\1 \2'),
    # 7. Fix broken abbreviations
    (re.compile(r'\b([a-z])\.\s+([a-z])\.'), r'\1.\2.'),
    # 8. Fix multiple spaces
    (re.compile(r' {2,}'), ' '),
    # 9. Fix broken parentheses and brackets
    (re.compile(r'\(\s+'), '('),
    (re.compile(r'\s+\)'), ')'),
    (re.compile(r'\[\s+'), '['),
    (re.compile(r'\s+\]'), ']'),
    # 10. Fix broken quotes
    (re.compile(r'"\s+([^"]+)\s+"'), r'"\1"'),
    (re.compile(r"'\s+([^']+)\s+'"), r"'\1'"),
    # 11. Normalize line breaks - improved pattern
    (re.compile(r'\n{3,}'), '\n\n'),
    # 12. Fix broken decimal numbers
    (re.compile(r'(\d)\s+\.\s+(\d)'), r'\1.\2'),
    # 13. Fix broken percentages
    (re.compile(r'(\d)\s+%'), r'\1%'),
    # 14. Fix broken units
    (re.compile(r'(\d)\s+([a-z]{1,3})\s+([a-z])'), r'\1 \2\3'),
    # 15. Fix broken LaTeX commands
    (re.compile(r'\\\s+([a-z]+)'), r'\\\1'),
    (re.compile(r'\\\s*\{'), r'\\{'),
    # 16. Fix broken equation numbers
    (re.compile(r'\((\d+)\)\s*$', re.MULTILINE), r'(\1)'),
    # 17. Fix broken figure/table references
    (re.compile(r'(Figure|Table|Fig\.|Tab\.)\s+(\d+)', re.IGNORECASE), r'\1 \2'),
    # 18. Fix broken section references
    (re.compile(r'Section\s+(\d+)', re.IGNORECASE), r'Section \1'),
    # 19. Fix broken equation references
    (re.compile(r'Equation\s+\((\d+)\)', re.IGNORECASE), r'Equation (\1)'),
    # 20. Fix broken dates
    (re.compile(r'(\d{4})\s+([A-Z][a-z]+)\s+(\d{1,2})'), r'\1 \2 \3'),
    # 21. Fix broken version numbers
    (re.compile(r'v\s*(\d+)'), r'v\1'),
    # 22. Fix broken page breaks in words - improved to be more conservative
    # Only join if lowercase letter followed by lowercase letter (broken word)
    (re.compile(r'\b([a-z])\s+\n\s+([a-z][a-z]+)'), r'\1\2'),
    # 23. Fix broken hyphenation at line breaks
    (re.compile(r'([a-zA-Z])-\s*\n\s*([a-zA-Z])'), r'\1\2'),
    # 24. Fix broken spaces in numbers
    (re.compile(r'(\d)\s+(\d{3})\b'), r'\1\2'),
    # 25. Fix broken special characters (HTML entities)
    (re.compile(r'&amp;'), '&'),
    (re.compile(r'&lt;'), '<'),
    (re.compile(r'&gt;'), '>'),
    (re.compile(r'&quot;'), '"'),
    # 26. Fix broken em/en dashes
    (re.compile(r'--'), '—'),
    (re.compile(r' - '), ' – '),
    # 27. Fix broken references and citations
    (re.compile(r'\[\s*(\d+)\s*\]'), r'[\1]'),
    (re.compile(r'\(\s*([A-Z][a-z]+)\s*,\s*(\d{4})\s*\)'), r'(\1, \2)'),
    # 28. Fix broken section numbers
    (re.compile(r'(\d+)\s*\.\s*(\d+)'), r'\1.\2'),
    # 29. Fix broken figure/table captions
    (re.compile(r'(Figure|Table|Fig\.|Tab\.)\s+(\d+)\s*:\s*', re.IGNORECASE), r'\1 \2: '),
    # 30. Fix broken list items
    (re.compile(r'^\s*(\d+)\s*\.\s+([A-Z])', re.MULTILINE), r'\1. \2'),
    (re.compile(r'^\s*[-•]\s+([A-Z])', re.MULTILINE), r'- \1'),
    # 31. Fix broken spacing in mathematical expressions
    (re.compile(r'([a-zA-Z0-9])\s*=\s*([a-zA-Z0-9])'), r'\1 = \2'),
    (re.compile(r'([a-zA-Z0-9])\s*<\s*([a-zA-Z0-9])'), r'\1 < \2'),
    (re.compile(r'([a-zA-Z0-9])\s*>\s*([a-zA-Z0-9])'), r'\1 > \2'),
    # 32. Fix broken equation references with spaces
    (re.compile(r'\((\s*\d+\s*)\)'), lambda m: '(' + m.group(1).strip() + ')'),
    # 33. Fix broken single-letter lines (initials, etc.) - preserve them but add proper spacing
    (re.compile(r'\n\s*([A-Z]\.)\s*\n'), r'\n\1\n'),
    # 34. Fix broken mathematical subscripts/superscripts
    (re.compile(r'([a-zA-Z0-9])_\s*\{?\s*([a-zA-Z0-9])\s*\}?'), r'\1_\2'),
    (re.compile(r'([a-zA-Z0-9])\^\s*\{?\s*([a-zA-Z0-9])\s*\}?'), r'\1^\2'),
)

# Patterns that need special handling (subscripts/superscripts - must run after simple patterns)
_POST_PATTERNS_SUBSUP = (
    (re.compile(r'([a-zA-Z0-9])\s*_\s*([a-zA-Z0-9])'), r'\1_\2'),
    (re.compile(r'([a-zA-Z0-9])\s*\^\s*([a-zA-Z0-9])'), r'\1^\2'),
)

# Multi-column fix pattern (lambda function)
_POST_MULTICOLUMN_PATTERN = re.compile(
    r'([a-zA-Z]{3,})\s+([a-zA-Z])\s+([a-zA-Z]{3,})'
)


class PDFExtractor:
    """Extract text from PDFs with multiple fallback methods."""
    
    def __init__(self, enable_ocr: bool = True, ocr_language: str = "eng", max_retries: int = 2,
                 enable_parallel: bool = True, max_workers: int = 4, large_pdf_threshold_mb: float = 20.0,
                 enable_caching: bool = True, cache_dir: Optional[str] = None,
                 max_memory_mb: float = 500.0, page_batch_size: int = 10):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        self.max_retries = max_retries
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.large_pdf_threshold_mb = large_pdf_threshold_mb
        self.enable_caching = enable_caching
        self.max_memory_mb = max_memory_mb
        self.page_batch_size = page_batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extraction_methods = []

        # Order of preference for extraction methods
        if PYMUPDF_AVAILABLE:
            self.extraction_methods.append(self._extract_pymupdf)
        if PDFPLUMBER_AVAILABLE:
            self.extraction_methods.append(self._extract_pdfplumber)
        if PYPDF_AVAILABLE:
            self.extraction_methods.append(self._extract_pypdf)

        if not self.extraction_methods:
            raise RuntimeError("No PDF extraction libraries available!")
    
    def _get_cache_key(self, pdf_path: Path) -> str:
        """Generate cache key from PDF file hash and modification time."""
        stat = pdf_path.stat()
        # Use file size and modification time for cache key
        key_data = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load extraction result from cache."""
        if not self.enable_caching:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    logger.debug(f"Loaded extraction from cache: {cache_key[:8]}")
                    return cached
            except Exception as e:
                logger.debug(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save extraction result to cache."""
        if not self.enable_caching:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Only cache successful extractions
            if result.get('success', False):
                # Cache the full extraction result including text and pages
                cached_result = {
                    'metadata': result.get('metadata', {}),
                    'text': result.get('text', ''),
                    'pages': result.get('pages', []),
                    'method_used': result.get('method_used'),
                    'quality_score': result.get('quality_score', 0.0),
                    'pdf_type': result.get('pdf_type', 'unknown'),
                    'num_pages': len(result.get('pages', [])),
                    'text_length': len(result.get('text', '')),
                    'cached': True
                }
                with open(cache_file, 'w') as f:
                    json.dump(cached_result, f, ensure_ascii=False)
                logger.debug(f"Saved extraction to cache: {cache_key[:8]}")
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def extract(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text and metadata from PDF with optimizations for large files.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with 'text', 'metadata', 'pages', 'method_used'
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        result = {
            'text': '',
            'metadata': {},
            'pages': [],
            'method_used': None,
            'success': False,
            'error': None
        }
        
        # Check cache first - return cached result immediately if available
        if self.enable_caching:
            cache_key = self._get_cache_key(pdf_path)
            cached = self._load_from_cache(cache_key)
            if cached:
                # Check if cache has full extraction (text and pages)
                if cached.get('text') and cached.get('pages') and cached.get('cached'):
                    logger.info(f"Using cached extraction for {pdf_path.name} (quality: {cached.get('quality_score', 0.0):.2f})")
                    result.update({
                        'text': cached.get('text', ''),
                        'metadata': cached.get('metadata', {}),
                        'pages': cached.get('pages', []),
                        'method_used': cached.get('method_used', 'cached'),
                        'quality_score': cached.get('quality_score', 0.0),
                        'pdf_type': cached.get('pdf_type', 'unknown'),
                        'success': True,
                        'cached': True
                    })
                    return result
                else:
                    # Legacy cache without full text - use metadata and continue extraction
                    logger.info(f"Found cached metadata for {pdf_path.name}, extracting text...")
                    if cached.get('metadata'):
                        result['metadata'] = cached.get('metadata')
                    if cached.get('pdf_type'):
                        result['pdf_type'] = cached.get('pdf_type')
        
        # Check PDF size for optimization
        pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        is_large_pdf = pdf_size_mb > self.large_pdf_threshold_mb

        if is_large_pdf:
            logger.debug(f"Large PDF detected ({pdf_size_mb:.1f} MB), checking memory for optimized extraction")

            # Check current memory usage and use memory-optimized extraction if needed
            current_memory = self._get_memory_usage_mb()
            if current_memory > 0 and current_memory > self.max_memory_mb * 0.5:
                logger.debug(f"Memory usage ({current_memory:.1f} MB) is high, using memory-optimized extraction")
                # Insert memory-optimized method at the beginning of the list
                self.extraction_methods.insert(0, self._extract_large_pdf_adaptive)
            elif pdf_size_mb > 100:
                # For very large PDFs, always use memory-optimized approach
                logger.debug(f"Very large PDF ({pdf_size_mb:.1f} MB), using memory-optimized extraction")
                self.extraction_methods.insert(0, self._extract_large_pdf_adaptive)
        
        # Detect PDF type (text-based vs scanned) to optimize extraction strategy
        pdf_type = self._detect_pdf_type(str(pdf_path))
        result['pdf_type'] = pdf_type
        
        # Optimize extraction strategy based on PDF type
        # For scanned PDFs, try OCR earlier
        if pdf_type == 'scanned' and self.enable_ocr:
            # Try OCR first for scanned PDFs
            try:
                logger.info(f"Detected scanned PDF, trying OCR first for {pdf_path.name}")
                extracted = self._extract_ocr(str(pdf_path))
                if extracted and self._validate_extraction(extracted):
                    ocr_score = self._score_extraction_quality(extracted)
                    if ocr_score >= 0.6:  # Acceptable quality for OCR
                        result.update(extracted)
                        result['success'] = True
                        result['quality_score'] = ocr_score
                        logger.info(f"Successfully extracted scanned PDF {pdf_path.name} using OCR (quality: {ocr_score:.2f})")
                        return result
            except Exception as e:
                logger.debug(f"Early OCR attempt failed: {e}")
        
        # Try each extraction method in order
        best_result = None
        best_score = 0
        
        for method in self.extraction_methods:
            # Retry logic for transient errors
            for attempt in range(self.max_retries + 1):
                try:
                    extracted = method(str(pdf_path))
                    if extracted and self._validate_extraction(extracted):
                        # Score the extraction quality
                        score = self._score_extraction_quality(extracted)
                        
                        # Keep the best extraction
                        if score > best_score:
                            best_score = score
                            best_result = extracted
                            
                        # Maximum quality mode: try all methods and keep the best
                        # Don't early exit - always try all extraction methods for maximum quality
                        pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
                        page_count = extracted.get('pages', [])
                        num_pages = len(page_count) if page_count else 0
                        is_very_large = pdf_size_mb > 50.0 or num_pages > 50
                        
                        # Maximum quality mode: always require high quality
                        if is_very_large:
                            quality_threshold = 0.90  # High quality even for very large PDFs
                        elif is_large_pdf:
                            quality_threshold = 0.92  # High quality for large PDFs
                        else:
                            quality_threshold = 0.95  # Maximum quality threshold
                        
                        # Continue trying all methods even if we hit threshold
                        # This ensures we get the absolute best extraction
                        if score >= quality_threshold and score >= best_score:
                            # Update best if this is better, but continue trying other methods
                            pass
                        # Don't break - continue trying all methods for maximum quality
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.debug(f"Extraction attempt {attempt + 1} failed for {method.__name__}, retrying...")
                        continue
                    else:
                        logger.warning(f"Extraction method {method.__name__} failed after {self.max_retries + 1} attempts: {e}")
                        break
                except KeyboardInterrupt:
                    # Allow interruption
                    raise
        
        # Use the best extraction found
        if best_result:
            result.update(best_result)
            result['success'] = True
            result['quality_score'] = best_score
            logger.info(f"Successfully extracted {pdf_path.name} using {result['method_used']} (quality: {best_score:.2f})")
            
            # Save to cache
            if self.enable_caching:
                cache_key = self._get_cache_key(pdf_path)
                self._save_to_cache(cache_key, result)
            
            return result
        
        # If we have a result but it's low quality, try to improve it
        if best_result and best_score < 0.7:
            # Try OCR for low-quality extractions (might be scanned)
            if self.enable_ocr:
                try:
                    logger.info(f"Low quality extraction ({best_score:.2f}), trying OCR for {pdf_path.name}")
                    extracted = self._extract_ocr(str(pdf_path))
                    if extracted and self._validate_extraction(extracted):
                        ocr_score = self._score_extraction_quality(extracted)
                        # Use OCR if it's significantly better
                        if ocr_score > best_score + 0.1:
                            result.update(extracted)
                            result['success'] = True
                            logger.info(f"OCR improved extraction quality to {ocr_score:.2f} for {pdf_path.name}")
                            return result
                except Exception as e:
                    logger.warning(f"OCR fallback failed: {e}")
        
        # Try OCR as last resort if no extraction succeeded
        if self.enable_ocr and result['text'] == '':
            try:
                extracted = self._extract_ocr(str(pdf_path))
                if extracted and self._validate_extraction(extracted):
                    result.update(extracted)
                    result['success'] = True
                    logger.info(f"Successfully extracted {pdf_path.name} using OCR")
                    return result
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                result['error'] = str(e)
        
        # If we have a best result (even if low quality), use it
        if best_result:
            result.update(best_result)
            result['success'] = True
            result['quality_warning'] = f"Low quality extraction (score: {best_score:.2f})"
            logger.warning(f"Using low-quality extraction for {pdf_path.name} (score: {best_score:.2f})")
            return result
        
        result['error'] = "All extraction methods failed"
        logger.error(f"Failed to extract text from {pdf_path.name}")
        return result
    
    def _extract_pymupdf(self, pdf_path: str) -> Optional[Dict]:
        """Extract using PyMuPDF (fitz) - best quality extraction with advanced methods and parallel processing."""
        doc = fitz.open(pdf_path)
        text_pages = []
        full_text = []
        
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc)
        }
        # Fix possible inversion of title/author (common in some PDFs)
        if metadata['title'] and metadata['author']:
            t = metadata['title'].lower()
            a = metadata['author'].lower()
            # Heuristic: title often contains "and" or "eds." while author looks like workshop/event name
            if (('and' in t or 'eds.' in t) and not ('and' in a or 'eds.' in a)):
                metadata['title'], metadata['author'] = metadata['author'], metadata['title']
        # Infer missing title/author from first page if still empty
        if not metadata['title'] or not metadata['author']:
            try:
                first_page = doc.load_page(0)
                page_text = first_page.get_text("text")
                lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
                if lines:
                    if not metadata['title']:
                        metadata['title'] = lines[0]
                    if len(lines) > 1 and not metadata['author']:
                        metadata['author'] = lines[1]
            except Exception as e:
                logger.debug(f"Failed to infer title/author from first page: {e}")
        # Infer missing title/author from first page if metadata empty
        if not metadata['title'] or not metadata['author']:
            try:
                first_page = doc.load_page(0)
                page_text = first_page.get_text("text")
                lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
                if lines:
                    # Heuristic: first line likely title, second line author
                    if not metadata['title']:
                        metadata['title'] = lines[0]
                    if len(lines) > 1 and not metadata['author']:
                        metadata['author'] = lines[1]
            except Exception as e:
                logger.debug(f"Failed to infer title/author from first page: {e}")
        
        # Check if we should use parallel processing
        # Optimized thresholds: >30 pages OR >30 MB (reduces overhead for smaller PDFs)
        pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
        current_memory = self._get_memory_usage_mb()

        # Use parallel for PDFs with >30 pages OR >30MB, but only if memory is not constrained
        memory_constrained = current_memory > 0 and current_memory > self.max_memory_mb * 0.6

        if self.enable_parallel and len(doc) > 30 and pdf_size_mb > 30.0 and not memory_constrained:
            # Parallel page extraction for large PDFs with sufficient memory
            text_pages = self._extract_pages_parallel(doc)
        elif memory_constrained or pdf_size_mb > 100:
            # Memory-constrained or very large PDFs - use batched extraction
            text_pages = self._extract_pages_batched(doc)
        else:
            # Sequential extraction for smaller PDFs
            text_pages = self._extract_pages_sequential(doc)
        
        # Combine page texts
        full_text = [page['text'] for page in text_pages]
        
        doc.close()
        
        # Post-process the full text for better quality
        full_text_combined = '\n\n'.join(full_text)
        full_text_combined = self._post_process_extracted_text(full_text_combined)
        
        return {
            'text': full_text_combined,
            'metadata': metadata,
            'pages': text_pages,
            'method_used': 'pymupdf'
        }
    
    def _extract_pages_sequential(self, doc) -> List[Dict]:
        """Extract pages sequentially (for smaller PDFs)."""
        text_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = self._extract_single_page(page, page_num)
            text_pages.append({
                'page': page_num + 1,
                'text': page_text or '',
                'char_count': len(page_text) if page_text else 0
            })
        return text_pages

    def _extract_pages_batched(self, doc) -> List[Dict]:
        """
        Extract pages in batches with memory management (for large/constrained PDFs).

        This method processes pages in small batches, forcing garbage collection
        between batches to control memory usage.
        """
        import gc

        text_pages = []
        total_pages = len(doc)

        # Determine batch size based on memory constraints
        pdf_size_mb = 0  # We don't have the path here, so use a default
        current_memory = self._get_memory_usage_mb()

        if current_memory > 0 and current_memory > self.max_memory_mb * 0.7:
            batch_size = 3
        elif total_pages > 100:
            batch_size = 5
        else:
            batch_size = self.page_batch_size

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)

            batch_pages = []
            for page_num in range(batch_start, batch_end):
                page = doc[page_num]
                page_text = self._extract_single_page(page, page_num)
                batch_pages.append({
                    'page': page_num + 1,
                    'text': page_text or '',
                    'char_count': len(page_text) if page_text else 0
                })

            text_pages.extend(batch_pages)

            # Force garbage collection between batches
            gc.collect()

            # Check memory and reduce batch size if needed
            if self._memory_exceeded() and batch_size > 1:
                logger.debug(f"Memory threshold exceeded, reducing batch size")

        return text_pages

    def _extract_pages_parallel(self, doc) -> List[Dict]:
        """Extract pages in parallel (for large PDFs)."""
        # Pre-extract page objects (PyMuPDF pages are safe to use across threads)
        pages = [doc[i] for i in range(len(doc))]
        text_pages = [None] * len(doc)
        
        def extract_page(page_data):
            page_num, page = page_data
            try:
                page_text = self._extract_single_page(page, page_num)
                return {
                    'page': page_num + 1,
                    'text': page_text or '',
                    'char_count': len(page_text) if page_text else 0,
                    'page_num': page_num
                }
            except Exception as e:
                logger.debug(f"Parallel extraction failed for page {page_num + 1}: {e}")
                return {
                    'page': page_num + 1,
                    'text': '',
                    'char_count': 0,
                    'page_num': page_num
                }
        
        # Use thread pool for parallel extraction
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(extract_page, (i, pages[i])): i for i in range(len(pages))}
            for future in as_completed(futures):
                result = future.result()
                text_pages[result['page_num']] = {
                    'page': result['page'],
                    'text': result['text'],
                    'char_count': result['char_count']
                }
        
        return text_pages

    # ============================================================
    # Memory-Optimized Extraction for Large PDFs
    # ============================================================

    def _extract_pymupdf_memory_optimized(self, pdf_path: str, callback=None) -> Optional[Dict]:
        """
        Memory-efficient extraction for very large PDFs.

        Processes pages in batches with memory monitoring to control
        memory usage during extraction of large documents.

        Args:
            pdf_path: Path to PDF file
            callback: Optional callback function(progress, total) for progress updates

        Returns:
            Extraction result dict or None if failed
        """
        import gc

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            result = {
                'text': '',
                'metadata': {},
                'pages': [],
                'method_used': 'pymupdf_memory_optimized',
                'success': True
            }

            # Extract metadata once
            result['metadata'] = self._extract_metadata(doc)

            # Dynamic batch size based on PDF size
            pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
            if pdf_size_mb > 100:
                batch_size = 5
            elif pdf_size_mb > 50:
                batch_size = 8
            else:
                batch_size = self.page_batch_size

            # Process in batches
            text_parts = []
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_pages = []

                for page_num in range(batch_start, batch_end):
                    page = doc[page_num]
                    page_text = self._extract_single_page(page, page_num)
                    page_data = {
                        'page': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text) if page_text else 0
                    }
                    batch_pages.append(page_data)

                    # Progress callback
                    if callback:
                        callback(page_num + 1, total_pages)

                # Add batch results
                result['pages'].extend(batch_pages)
                text_parts.append('\n\n'.join(p['text'] for p in batch_pages))

                # Force garbage collection between batches
                gc.collect()

                # Memory monitoring and adaptive batch sizing
                if self._memory_exceeded():
                    logger.warning(f"Memory threshold exceeded at page {batch_end}, reducing batch size")
                    batch_size = max(1, batch_size // 2)

            # Combine all text
            result['text'] = '\n\n'.join(text_parts)

            # Post-process the combined text
            result['text'] = self._post_process_extracted_text(result['text'])

            doc.close()
            return result

        except Exception as e:
            logger.warning(f"Memory-optimized extraction failed: {e}")
            return None

    def _memory_exceeded(self) -> bool:
        """
        Check if current process memory usage exceeds threshold.

        Returns:
            True if memory exceeds max_memory_mb, False otherwise
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb > self.max_memory_mb
        except ImportError:
            # If psutil not available, use a simpler check
            # Count accumulated strings as a proxy for memory
            return False

    def _get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes, or -1 if unavailable
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1

    def _extract_large_pdf_adaptive(self, pdf_path: str, callback=None) -> Optional[Dict]:
        """
        Adaptively extract large PDFs using the best available method.

        This method:
        1. Checks available memory
        2. Chooses between parallel and batched extraction
        3. Monitors memory during extraction
        4. Falls back to batched extraction if memory is low

        Args:
            pdf_path: Path to PDF file
            callback: Optional callback for progress updates

        Returns:
            Extraction result dict
        """
        pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
        current_memory = self._get_memory_usage_mb()

        # If memory is already high, use memory-optimized approach
        if current_memory > 0 and current_memory > self.max_memory_mb * 0.7:
            logger.info(f"High memory usage ({current_memory:.1f} MB), using memory-optimized extraction")
            return self._extract_pymupdf_memory_optimized(pdf_path, callback)

        # If PDF is very large, also use memory-optimized approach
        if pdf_size_mb > 100:
            logger.info(f"Very large PDF ({pdf_size_mb:.1f} MB), using memory-optimized extraction")
            return self._extract_pymupdf_memory_optimized(pdf_path, callback)

        # Otherwise, use standard extraction (which already has parallel optimization)
        return self._extract_pymupdf(pdf_path)

    def _stream_extract_pymupdf(self, pdf_path: str, chunk_size: int = 50,
                                  process_callback=None) -> Dict:
        """
        Stream-based extraction for extremely large PDFs.

        Yields extracted content in chunks instead of accumulating all in memory.
        Useful for PDFs that don't fit in memory even with batched extraction.

        Args:
            pdf_path: Path to PDF file
            chunk_size: Number of pages per chunk
            process_callback: Optional callback(chunk_index, total_chunks, chunk_data)

        Yields:
            Dict containing 'chunk_index', 'total_chunks', 'text', 'pages'
        """
        import gc

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        total_chunks = (total_pages + chunk_size - 1) // chunk_size

        metadata = self._extract_metadata(doc)

        for chunk_idx in range(total_chunks):
            start_page = chunk_idx * chunk_size
            end_page = min(start_page + chunk_size, total_pages)

            chunk_pages = []
            chunk_text_parts = []

            for page_num in range(start_page, end_page):
                page = doc[page_num]
                page_text = self._extract_single_page(page, page_num)
                page_data = {
                    'page': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text) if page_text else 0
                }
                chunk_pages.append(page_data)
                chunk_text_parts.append(page_text)

            chunk_text = '\n\n'.join(chunk_text_parts)
            chunk_data = {
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'text': chunk_text,
                'pages': chunk_pages,
                'metadata': metadata if chunk_idx == 0 else {}
            }

            # Process callback if provided
            if process_callback:
                process_callback(chunk_idx, total_chunks, chunk_data)

            yield chunk_data

            # Force garbage collection between chunks
            gc.collect()

        doc.close()

    def _extract_single_page(self, page, page_num: int) -> str:
        """Extract text from a single page with multiple strategies including tables."""
        # Strategy 1: Try dict-based extraction for structured text (best quality)
        page_text = None
        try:
            text_dict = page.get_text("dict")
            if text_dict and text_dict.get('blocks'):
                # Reconstruct text from dict with proper ordering
                page_text = self._reconstruct_text_from_dict(text_dict)
        except Exception as e:
            logger.debug(f"Dict extraction failed for page {page_num + 1}: {e}")
        
        # Strategy 2: Layout-preserving extraction (better for multi-column)
        # Always try layout extraction for maximum quality
        try:
            page_text_layout = page.get_text("layout")
            if page_text_layout:
                # Use layout if it's significantly better or if current text is incomplete
                if not page_text or len(page_text_layout) > len(page_text) * 1.02 or len(page_text) < 1000:
                    page_text = page_text_layout
        except Exception as e:
            logger.debug(f"Layout extraction failed, falling back: {e}")
        
        # Strategy 3: Standard extraction as fallback
        if not page_text:
            try:
                page_text = page.get_text()
            except Exception as e:
                logger.debug(f"Standard text extraction failed: {e}")
                page_text = ''
        
        # Strategy 4: Extract tables using PyMuPDF (if available)
        # Try to find and extract tables to enhance content
        # Note: Disabled to avoid find_tables stderr noise (harmless but cluttered logs)
        # Tables are still extracted via text blocks, just not using find_tables API
        # try:
        #     tables = self._extract_tables_pymupdf(page)
        #     if tables:
        #         table_text = self._format_tables_markdown(tables)
        #         if table_text:
        #             if page_text:
        #                 page_text += '\n\n' + table_text
        #             else:
        #                 page_text = table_text
        # except Exception as e:
        #     logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
        
        # Strategy 5: Text blocks with coordinates (for better multi-column handling)
        # Always try blocks for maximum quality (better multi-column and table handling)
        try:
            blocks = page.get_text("blocks")
            if blocks:
                page_text_blocks = self._reconstruct_text_from_blocks(blocks)
                # Use blocks if they provide more text or better structure
                if page_text_blocks and (len(page_text_blocks) > len(page_text) * 0.95 or 
                                         (len(page_text_blocks) > len(page_text) * 0.8 and len(page_text) < 2000)):
                    page_text = page_text_blocks
        except Exception as e:
            logger.debug(f"Block extraction failed, skipping: {e}")
        
        # Strategy 6: Try rawdict for very structured documents (always try for maximum quality)
        if len(page_text) < 1000:  # Try rawdict if text seems incomplete
            try:
                rawdict = page.get_text("rawdict")
                if rawdict and rawdict.get('blocks'):
                    page_text_raw = self._reconstruct_text_from_dict(rawdict)
                    if page_text_raw and len(page_text_raw) > len(page_text) * 0.95:
                        page_text = page_text_raw
            except Exception as e:
                logger.debug(f"Rawdict extraction failed, skipping: {e}")
        
        return page_text
    
    def _extract_tables_pymupdf(self, page) -> List[List[List[str]]]:
        """
        Extract tables from a PyMuPDF page using enhanced table detection.
        
        This method:
        1. Uses PyMuPDF's find_tables if available
        2. Falls back to block-based detection for complex layouts
        3. Handles rotated and scanned tables
        4. Preserves table structure across page breaks
        """
        import sys
        import io
        import os
        from contextlib import redirect_stderr
        
        tables = []
        
        try:
            # Method PyMuPDF's 1: Try find_tables (best quality)
            if hasattr(page, 'find_tables'):
                with open(os.devnull, 'w') as devnull:
                    with redirect_stderr(devnull):
                        try:
                            table_list = page.find_tables()
                            for table in table_list:
                                try:
                                    table_data = table.extract()
                                    if table_data and len(table_data) > 0:
                                        # Clean and validate extracted data
                                        cleaned_data = self._clean_table_data(table_data)
                                        if cleaned_data:
                                            tables.append(cleaned_data)
                                except Exception as e:
                                    logger.debug(f"Table data extraction failed: {e}")
                                    continue
                        except Exception as e:
                            logger.debug(f"find_tables method failed: {e}")
            
            # Method 2: If no tables found, try block-based detection
            if not tables:
                blocks = page.get_text("blocks")
                if blocks:
                    table_blocks = self._detect_table_blocks_enhanced(blocks, page)
                    if table_blocks:
                        tables.extend(table_blocks)
            
            # Method 3: Try to detect tables from lines (for scanned PDFs)
            if not tables:
                lines = page.get_text("lines")
                if lines:
                    line_tables = self._detect_tables_from_lines(lines)
                    if line_tables:
                        tables.extend(line_tables)
                        
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
        
        return tables
    
    def _clean_table_data(self, table_data: List[List[str]]) -> Optional[List[List[str]]]:
        """
        Clean and validate extracted table data.
        
        Removes empty rows/columns, normalizes spacing, etc.
        """
        if not table_data:
            return None
        
        cleaned = []
        
        for row in table_data:
            if not row:
                continue
            
            # Clean each cell
            cleaned_row = []
            has_content = False
            
            for cell in row:
                if cell is None:
                    cell = ''
                else:
                    cell = str(cell).strip()
                
                cleaned_row.append(cell)
                
                if cell:
                    has_content = True
            
            # Only keep rows with some content
            if has_content:
                cleaned.append(cleaned_row)
        
        # Check if table has enough data
        if len(cleaned) < 2 or len(cleaned[0]) < 2:
            return None
        
        return cleaned
    
    def _detect_table_blocks_enhanced(self, blocks: List, page) -> List[List[List[str]]]:
        """
        Enhanced table detection from text blocks.
        
        Uses multiple heuristics:
        1. Aligned blocks (same y position for columns)
        2. Consistent spacing between columns
        3. Table-like separators (lines, dashes)
        4. Density of content in rectangular regions
        """
        if not blocks:
            return []
        
        tables = []
        
        # Filter text blocks only
        text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip()]
        
        if len(text_blocks) < 6:  # Need minimum blocks for a table
            return []
        
        # Group blocks by vertical position (potential rows)
        rows = self._group_blocks_into_rows(text_blocks)
        
        # Check if grouped blocks form a table structure
        if len(rows) < 2:
            return []
        
        # Detect columns from aligned blocks
        columns = self._detect_columns_from_blocks(text_blocks)
        
        if len(columns) < 2:
            return []
        
        # Convert to table format
        table = self._blocks_to_table(rows, columns, text_blocks)
        
        if table and self._is_valid_table(table):
            tables.append(table)
        
        return tables
    
    def _group_blocks_into_rows(self, blocks: List, tolerance: float = 8.0) -> Dict[float, List]:
        """
        Group blocks into rows based on y-coordinate similarity.
        
        Args:
            blocks: List of PDF blocks
            tolerance: Y-position tolerance for same row (in points)
            
        Returns:
            Dict mapping y-position (rounded) to list of blocks
        """
        rows = {}
        
        for block in blocks:
            if len(block) < 2:
                continue
            
            y_pos = block[1]  # y0 coordinate
            row_key = round(y_pos / tolerance) * tolerance
            
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(block)
        
        return rows
    
    def _detect_columns_from_blocks(self, blocks: List) -> List[float]:
        """
        Detect column positions from block x-coordinates.
        
        Uses clustering to find consistent column positions.
        """
        if not blocks:
            return []
        
        # Get center x positions
        centers = []
        for block in blocks:
            if len(block) >= 4:
                center_x = (block[0] + block[2]) / 2
                centers.append(center_x)
        
        if len(centers) < 3:
            return []
        
        # Sort and find gaps
        centers.sort()
        
        # Calculate gaps between consecutive centers
        gaps = []
        for i in range(len(centers) - 1):
            gap = centers[i + 1] - centers[i]
            avg_pos = (centers[i] + centers[i + 1]) / 2
            gaps.append((avg_pos, gap))
        
        # Find significant gaps (potential column separators)
        if not gaps:
            return []
        
        avg_gap = sum(g[1] for g in gaps) / len(gaps)
        threshold = avg_gap * 2.5
        
        column_positions = [centers[0]]  # First column starts at first center
        
        for pos, gap in gaps:
            if gap > threshold:
                # This is a column separator
                column_positions.append(pos + gap / 2)
        
        column_positions.append(centers[-1])  # Last column ends at last center
        
        return column_positions
    
    def _blocks_to_table(self, rows: Dict[float, List], columns: List[float], 
                         all_blocks: List) -> List[List[str]]:
        """
        Convert grouped blocks and columns to table format.
        """
        if not rows or not columns:
            return []
        
        # Sort rows by y position (top to bottom)
        sorted_row_keys = sorted(rows.keys())
        
        # Create table grid
        table = []
        
        for row_key in sorted_row_keys:
            row_blocks = rows[row_key]
            row_data = [''] * (len(columns) - 1)
            
            for block in row_blocks:
                if len(block) < 5:
                    continue
                
                block_text = block[4].strip()
                if not block_text:
                    continue
                
                # Find which column this block belongs to
                block_center = (block[0] + block[2]) / 2
                
                for i in range(len(columns) - 1):
                    if columns[i] <= block_center < columns[i + 1]:
                        if row_data[i]:
                            row_data[i] += ' ' + block_text
                        else:
                            row_data[i] = block_text
                        break
            
            # Only add non-empty rows
            if any(cell for cell in row_data):
                table.append(row_data)
        
        return table
    
    def _is_valid_table(self, table: List[List[str]]) -> bool:
        """
        Check if extracted data forms a valid table.
        """
        if not table or len(table) < 2:
            return False
        
        # Check row consistency
        first_row_len = len(table[0])
        for row in table:
            if len(row) != first_row_len:
                return False
            # Check if row has meaningful content
            non_empty = sum(1 for cell in row if cell)
            if non_empty == 0:
                return False
        
        # Check minimum size
        if len(table) < 2 or first_row_len < 2:
            return False
        
        return True
    
    def _detect_tables_from_lines(self, lines: List) -> List[List[List[str]]]:
        """
        Detect tables from PDF lines (useful for scanned PDFs).
        
        Looks for:
        1. Horizontal lines that might be table separators
        2. Aligned text blocks that form rows
        3. Consistent vertical spacing
        """
        if not lines:
            return []
        
        tables = []
        
        # Extract horizontal lines
        horizontal_lines = []
        for line in lines:
            if len(line) >= 4:
                x0, y0, x1, y1 = line[:4]
                # Check if it's approximately horizontal
                if y1 - y0 < 2 and x1 - x0 > 20:  # Thin horizontal line
                    horizontal_lines.append({
                        'y0': y0,
                        'y1': y1,
                        'x0': x0,
                        'x1': x1,
                        'length': x1 - x0
                    })
        
        if len(horizontal_lines) < 2:
            return []
        
        # Sort lines by y position
        horizontal_lines.sort(key=lambda l: l['y0'])
        
        # Find groups of lines that might be table borders
        table_regions = self._find_table_regions(horizontal_lines)
        
        for region in table_regions:
            table = self._extract_table_from_region(region, lines)
            if table and self._is_valid_table(table):
                tables.append(table)
        
        return tables
    
    def _find_table_regions(self, lines: List[Dict]) -> List[Dict]:
        """
        Find regions bounded by horizontal lines that might contain tables.
        """
        if len(lines) < 2:
            return []
        
        regions = []
        
        # Look for pairs of lines that could be top/bottom borders
        for i in range(len(lines) - 1):
            top_line = lines[i]
            bottom_line = lines[i + 1]
            
            # Check if lines have similar length (potential table borders)
            length_ratio = min(top_line['length'], bottom_line['length']) / max(top_line['length'], bottom_line['length'])
            
            if length_ratio > 0.7:  # Similar length
                # Check if they're close together (not a full page)
                vertical_gap = bottom_line['y0'] - top_line['y1']
                
                if vertical_gap > 10 and vertical_gap < 500:  # Reasonable table height
                    regions.append({
                        'top_y': top_line['y0'],
                        'bottom_y': bottom_line['y1'],
                        'left_x': min(top_line['x0'], bottom_line['x0']),
                        'right_x': max(top_line['x1'], bottom_line['x1']),
                        'top_line': top_line,
                        'bottom_line': bottom_line
                    })
        
        return regions
    
    def _extract_table_from_region(self, region: Dict, lines: List) -> List[List[str]]:
        """
        Extract table content from a detected region.
        """
        # This is a simplified extraction - real implementation would need
        # more sophisticated analysis of text positions within the region
        
        table = []
        
        # Find text lines within the region
        for line in lines:
            if len(line) >= 5 and isinstance(line[4], str):
                line_text = line[4].strip()
                y_pos = line[1]
                
                # Check if line is within region
                if region['top_y'] <= y_pos <= region['bottom_y']:
                    # Check if line looks like table content
                    # (contains separators, aligned content, etc.)
                    if self._is_table_row(line_text, lines, y_pos):
                        row = self._parse_table_row(line_text)
                        if row:
                            table.append(row)
        
        return table
    
    def _is_table_row(self, line_text: str, all_lines: List, y_pos: float) -> bool:
        """
        Check if a line looks like a table row.
        """
        # Table rows often contain separators
        if '|' in line_text:
            return True
        
        # Check for tab-separated or aligned content
        # This is a simplified check
        
        # Check if nearby lines have similar structure
        nearby_lines = [l for l in all_lines if len(l) >= 2 and abs(l[1] - y_pos) < 15]
        
        if len(nearby_lines) >= 2:
            # Check if lines have similar character counts (indicates columns)
            char_counts = [len(str(l[4]) if len(l) > 4 else '') for l in nearby_lines if len(l) > 4]
            if len(char_counts) >= 2:
                count_variance = max(char_counts) - min(char_counts)
                if count_variance < 10:  # Similar widths
                    return True
        
        return False
    
    def _parse_table_row(self, line_text: str) -> List[str]:
        """
        Parse a table row into cells.
        """
        # Handle pipe-separated tables
        if '|' in line_text:
            cells = line_text.split('|')
            return [cell.strip() for cell in cells if cell.strip()]
        
        # Handle tab-separated
        if '\t' in line_text:
            cells = line_text.split('\t')
            return [cell.strip() for cell in cells if cell.strip()]
        
        # Check for comma-separated (CSV-like)
        if ',' in line_text and not (' and ' in line_text.lower() or ' or ' in line_text.lower()):
            cells = line_text.split(',')
            if len(cells) >= 2:
                return [cell.strip() for cell in cells if cell.strip()]
        
        return None
    
    def _format_tables_markdown(self, tables: List[List[List[str]]], 
                                 include_metadata: bool = True) -> str:
        """
        Format extracted tables as enhanced Markdown for better readability.
        
        Args:
            tables: List of 2D table data
            include_metadata: Include table number and cell count
            
        Returns:
            Formatted Markdown string
        """
        if not tables:
            return ''
        
        formatted_tables = []
        
        for table_idx, table in enumerate(tables, 1):
            if not table or len(table) == 0:
                continue
            
            table_lines = []
            
            # Add table header
            if include_metadata:
                num_rows = len(table)
                num_cols = max(len(row) for row in table) if table else 0
                table_lines.append(f'\n### Table {table_idx} ({num_rows} rows × {num_cols} columns)\n')
            
            # Determine number of columns
            max_cols = max(len(row) for row in table if row) if table else 0
            if max_cols == 0:
                continue
            
            # Detect if first row is a header
            has_header = self._detect_table_header(table)
            
            # Format rows
            for row_idx, row in enumerate(table):
                # Pad row to max_cols
                padded_row = row + [''] * (max_cols - len(row))
                
                # Clean and format cells
                cells = []
                for cell in padded_row:
                    cell_str = str(cell).strip() if cell else ''
                    # Escape Markdown special chars
                    cell_str = cell_str.replace('|', '\\|')
                    cell_str = cell_str.replace('*', '\\*')
                    cell_str = cell_str.replace('_', '\\_')
                    cell_str = cell_str.replace('`', '\\`')
                    cells.append(cell_str)
                
                if row_idx == 0 and has_header:
                    # Format header row with bold
                    header_cells = ['**' + cell + '**' if cell else '' for cell in cells]
                    table_lines.append('| ' + ' | '.join(header_cells) + ' |')
                else:
                    table_lines.append('| ' + ' | '.join(cells) + ' |')
                
                # Add separator after header
                if row_idx == 0 and has_header:
                    separator = ['---'] * max_cols
                    table_lines.append('| ' + ' | '.join(separator) + ' |')
            
            formatted_tables.append('\n'.join(table_lines))
        
        return '\n\n'.join(formatted_tables)
    
    def _detect_table_header(self, table: List[List[str]]) -> bool:
        """
        Detect if the first row of a table is a header.
        
        Heuristics:
        1. First row is shorter than others
        2. First row contains fewer numbers
        3. First row has shorter average cell length
        """
        if len(table) < 2:
            return False
        
        first_row = table[0]
        other_rows = table[1:]
        
        # Check if first row is notably shorter
        if len(first_row) < len(table[1]) * 0.7:
            return True
        
        # Check for number content
        first_row_numbers = sum(1 for cell in first_row if self._contains_number(cell))
        other_numbers = sum(
            sum(1 for cell in row if self._contains_number(cell))
            for row in other_rows[:5]  # Check first 5 rows
        ) / min(len(other_rows), 5)
        
        if other_rows:
            other_avg_numbers = sum(
                sum(1 for cell in row if self._contains_number(cell))
                for row in other_rows
            ) / len(other_rows)
            
            # If first row has notably fewer numbers, likely a header
            if other_avg_numbers > 0 and first_row_numbers < other_avg_numbers * 0.3:
                return True
        
        # Check average cell length
        if first_row and other_rows:
            first_avg_len = sum(len(str(cell)) for cell in first_row) / len(first_row)
            other_avg_len = sum(
                sum(len(str(cell)) for cell in row) / len(row)
                for row in other_rows
            ) / len(other_rows)
            
            # Headers often have shorter cell text
            if first_avg_len < other_avg_len * 0.7:
                return True
        
        return False
    
    def _contains_number(self, text: str) -> bool:
        """Check if text contains numeric content."""
        if not text:
            return False
        return bool(re.search(r'\d', str(text)))
    
    def _format_table_for_latex(self, tables: List[List[List[str]]]) -> str:
        """
        Format tables for LaTeX output.
        """
        if not tables:
            return ''
        
        latex_tables = []
        
        for table_idx, table in enumerate(tables, 1):
            if not table or len(table) == 0:
                continue
            
            latex_parts = []
            latex_parts.append(f'% Table {table_idx}')
            latex_parts.append(r'\begin{tabular}')
            
            # Determine column format
            num_cols = max(len(row) for row in table if row)
            col_format = '|' + 'l|' * num_cols
            latex_parts.append(f'{{{col_format}}}')
            
            # Add header separator
            latex_parts.append(r'\hline')
            
            # Format rows
            for row_idx, row in enumerate(table):
                padded_row = row + [''] * (num_cols - len(row))
                cells = [str(cell).replace('_', '\\_') if cell else '' for cell in padded_row]
                latex_parts.append(' & '.join(cells) + r' \\')
                latex_parts.append(r'\hline')
            
            latex_parts.append(r'\end{tabular}')
            latex_tables.append('\n'.join(latex_parts))
        
        return '\n\n'.join(latex_tables)
    
    def _merge_tables_across_pages(self, tables: List[List[List[str]]], 
                                    page_breaks: List[int]) -> List[List[List[str]]]:
        """
        Merge tables that span across multiple pages.
        
        Args:
            tables: List of tables from consecutive pages
            page_breaks: Indices indicating page boundaries
            
        Returns:
            Merged tables
        """
        if len(tables) <= 1:
            return tables
        
        merged_tables = []
        current_table = []
        
        for i, table in enumerate(tables):
            if not table:
                continue
            
            # Check if this looks like a continuation of previous table
            if current_table:
                # Check if first row looks like a continuation
                if self._is_table_continuation(current_table, table):
                    # Merge tables
                    merged = self._concatenate_tables(current_table, table)
                    current_table = merged
                else:
                    # Different table - save current and start new
                    merged_tables.append(current_table)
                    current_table = table
            else:
                current_table = table
        
        # Don't forget the last table
        if current_table:
            merged_tables.append(current_table)
        
        return merged_tables
    
    def _is_table_continuation(self, table1: List[List[str]], 
                               table2: List[List[str]]) -> bool:
        """
        Check if table2 looks like a continuation of table1.
        """
        if not table1 or not table2:
            return False
        
        # Check column consistency
        if len(table1[0]) != len(table2[0]):
            return False
        
        # Check if table2 starts without a header (typical for continuation)
        if len(table2) >= 2:
            # If table2 has similar structure to table1's data rows
            # and doesn't have header-like first row
            if not self._detect_table_header(table2):
                return True
        
        return False
    
    def _concatenate_tables(self, table1: List[List[str]], 
                            table2: List[List[str]]) -> List[List[str]]:
        """
        Concatenate two tables (table2 follows table1).
        """
        # Skip header if table2 has one
        start_idx = 1 if self._detect_table_header(table2) else 0
        return table1 + table2[start_idx:]
    
    # ============================================================
    # Reference Section Special Handling
    # ============================================================
    
    def _detect_reference_section(self, text: str) -> Dict:
        """
        Detect and extract the reference section from text.
        
        Returns:
            Dict with:
            - has_references: bool indicating if reference section exists
            - start_position: character position where references start
            - reference_text: the extracted reference section
            - reference_format: 'numbered', 'author-year', 'bibtex', or 'unknown'
        """
        if not text:
            return {'has_references': False}
        
        lines = text.split('\n')
        
        # Reference section detection patterns
        ref_section_patterns = [
            r'^\s*references?\s*$',
            r'^\s*bibliography\s*$',
            r'^\s*references?\s*\[',
            r'^\s*references?\s*:\s*$',
            r'^\s*bibliographic\s*references?\s*$',
            r'^\s*literature\s*references?\s*$',
        ]
        
        # Find reference section start
        ref_start = -1
        ref_pattern_used = None
        
        for i, line in enumerate(lines):
            for pattern in ref_section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    ref_start = i
                    ref_pattern_used = pattern
                    break
            if ref_start >= 0:
                break
        
        if ref_start < 0:
            return {'has_references': False}
        
        # Extract reference section
        ref_lines = lines[ref_start:]
        ref_text = '\n'.join(ref_lines)
        
        # Detect reference format
        ref_format = self._detect_reference_format(ref_text)
        
        return {
            'has_references': True,
            'start_position': ref_start,
            'reference_text': ref_text,
            'reference_format': ref_format,
            'section_header': lines[ref_start].strip() if ref_start < len(lines) else ''
        }
    
    def _detect_reference_format(self, ref_text: str) -> str:
        """
        Detect the format of references.
        
        Formats:
        - 'numbered': [1] Smith et al. (2020)
        - 'author-year': Smith (2020) showed that...
        - 'bibtex': @article{...}
        - 'unknown': could not determine
        """
        # Check for BibTeX format
        if '@' in ref_text and ('@article' in ref_text.lower() or 
                                 '@inproceedings' in ref_text.lower() or
                                 '@book' in ref_text.lower()):
            return 'bibtex'
        
        # Check for numbered references (most common)
        numbered_patterns = [
            r'^\s*\[\d+\]\s+',
            r'^\s*\[\d+\.\d+\]\s+',
            r'^\s*\d+\.\s+[A-Z]',
        ]
        
        lines = ref_text.split('\n')[:10]  # Check first 10 lines
        numbered_count = 0
        
        for line in lines:
            line = line.strip()
            for pattern in numbered_patterns:
                if re.match(pattern, line):
                    numbered_count += 1
                    break
        
        if numbered_count >= 3:  # At least 3 numbered references
            return 'numbered'
        
        # Check for author-year format
        author_year_patterns = [
            r'^[A-Z][a-z]+\s+\(\d{4}\)',
            r'^[A-Z][A-Z]+\s+et\s+al\.\s+\(\d{4}\)',
        ]
        
        author_year_count = 0
        for line in lines:
            line = line.strip()
            for pattern in author_year_patterns:
                if re.match(pattern, line):
                    author_year_count += 1
                    break
        
        if author_year_count >= 3:
            return 'author-year'
        
        return 'unknown'
    
    def _parse_references(self, text: str) -> Dict:
        """
        Parse references from text and return structured data.
        
        Returns:
            Dict with:
            - references: list of parsed reference dicts
            - reference_section: the extracted reference text
            - format: detected reference format
            - identifiers: extracted DOIs, arXiv IDs, URLs
        """
        import re
        
        result = {
            'references': [],
            'reference_section': '',
            'format': 'unknown',
            'identifiers': {
                'dois': [],
                'arxiv_ids': [],
                'urls': []
            }
        }
        
        # Detect reference section
        ref_info = self._detect_reference_section(text)
        
        if not ref_info.get('has_references', False):
            return result
        
        result['reference_section'] = ref_info.get('reference_text', '')
        result['format'] = ref_info.get('reference_format', 'unknown')
        
        ref_text = ref_info['reference_text']
        
        # Parse based on format
        if result['format'] == 'numbered':
            result['references'] = self._parse_numbered_references(ref_text)
        elif result['format'] == 'author-year':
            result['references'] = self._parse_author_year_references(ref_text)
        elif result['format'] == 'bibtex':
            result['references'] = self._parse_bibtex_references(ref_text)
        else:
            # Try all parsers and combine results
            all_refs = []
            all_refs.extend(self._parse_numbered_references(ref_text))
            all_refs.extend(self._parse_author_year_references(ref_text))
            all_refs.extend(self._parse_bibtex_references(ref_text))
            result['references'] = all_refs
        
        # Extract identifiers from all references
        for ref in result['references']:
            result['identifiers']['dois'].extend(ref.get('dois', []))
            result['identifiers']['arxiv_ids'].extend(ref.get('arxiv_ids', []))
            result['identifiers']['urls'].extend(ref.get('urls', []))
        
        # Remove duplicates
        for key in result['identifiers']:
            result['identifiers'][key] = list(set(result['identifiers'][key]))
        
        return result
    
    def _parse_numbered_references(self, ref_text: str) -> List[Dict]:
        """
        Parse numbered references [1] Smith et al. (2020).
        """
        import re
        
        references = []
        
        # Split references by numbering pattern
        # Pattern: [N] or [N.M] at start of line
        ref_pattern = r'^\s*\[(\d+(?:\.\d+)?)\]\s*(.+)$'
        
        lines = ref_text.split('\n')
        current_ref = None
        current_number = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(ref_pattern, line, re.MULTILINE)
            
            if match:
                # Save previous reference
                if current_ref and current_number:
                    parsed = self._parse_reference_content(current_ref, current_number, 'numbered')
                    if parsed:
                        references.append(parsed)
                
                # Start new reference
                current_number = match.group(1)
                current_ref = match.group(2)
            else:
                # Continuation of previous reference
                if current_ref:
                    current_ref += ' ' + line
        
        # Don't forget the last reference
        if current_ref and current_number:
            parsed = self._parse_reference_content(current_ref, current_number, 'numbered')
            if parsed:
                references.append(parsed)
        
        return references
    
    def _parse_author_year_references(self, ref_text: str) -> List[Dict]:
        """
        Parse author-year references Smith (2020).
        """
        import re
        
        references = []
        
        # Split by common patterns
        # Pattern: Author (Year) or Author et al. (Year)
        ref_pattern = r'^([A-Z][a-zA-Z\s\.]+?)\s*\((\d{4})\)\.?\s*(.+)$'
        
        lines = ref_text.split('\n')
        current_ref = None
        current_authors = None
        current_year = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(ref_pattern, line)
            
            if match:
                # Save previous reference
                if current_ref and current_authors:
                    parsed = self._parse_reference_content(
                        current_ref, 
                        f"{current_authors} ({current_year})", 
                        'author-year'
                    )
                    if parsed:
                        references.append(parsed)
                
                # Start new reference
                current_authors = match.group(1).strip()
                current_year = match.group(2)
                current_ref = match.group(3)
            else:
                # Continuation of previous reference
                if current_ref:
                    current_ref += ' ' + line
        
        # Don't forget the last reference
        if current_ref and current_authors:
            parsed = self._parse_reference_content(
                current_ref,
                f"{current_authors} ({current_year})",
                'author-year'
            )
            if parsed:
                references.append(parsed)
        
        return references
    
    def _parse_bibtex_references(self, ref_text: str) -> List[Dict]:
        """
        Parse BibTeX format references @article{...}.
        """
        import re
        
        references = []
        
        # Find BibTeX entries
        entry_pattern = r'@(\w+)\s*\{\s*([^,]+),'
        
        entries = list(re.finditer(entry_pattern, ref_text, re.IGNORECASE))
        
        for i, entry in enumerate(entries):
            entry_type = entry.group(1)
            entry_key = entry.group(2)
            
            # Find entry content
            start_pos = entry.end()
            end_pos = ref_text.find('@', start_pos)
            
            if end_pos == -1:
                end_pos = len(ref_text)
            
            entry_content = ref_text[start_pos:end_pos]
            
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
            
            for field_match in re.finditer(field_pattern, entry_content, re.IGNORECASE):
                field_name = field_match.group(1).lower()
                field_value = field_match.group(2).strip()
                fields[field_name] = field_value
            
            ref = {
                'type': entry_type,
                'key': entry_key,
                'title': fields.get('title', ''),
                'authors': fields.get('author', ''),
                'year': fields.get('year', ''),
                'journal': fields.get('journal', fields.get('booktitle', '')),
                'doi': fields.get('doi', ''),
                'arxiv_id': fields.get('arxiv', fields.get('eprint', '')),
                'url': fields.get('url', ''),
                'raw': f"@{entry_type}{{{entry_key},{entry_content[:200]}...}}"
            }
            
            # Extract additional identifiers
            ref['dois'] = self._extract_dois_from_text(ref.get('raw', ''))
            ref['arxiv_ids'] = self._extract_arxiv_ids_from_text(ref.get('raw', ''))
            ref['urls'] = self._extract_urls_from_text(ref.get('raw', ''))
            
            references.append(ref)
        
        return references
    
    def _parse_reference_content(self, content: str, citation: str, 
                                  ref_format: str) -> Optional[Dict]:
        """
        Parse a single reference into structured format.
        """
        import re
        
        ref = {
            'citation': citation,
            'format': ref_format,
            'raw': content,
            'authors': '',
            'year': '',
            'title': '',
            'journal': '',
            'dois': [],
            'arxiv_ids': [],
            'urls': []
        }
        
        # Extract year
        year_match = re.search(r'\((\d{4})\)', content)
        if year_match:
            ref['year'] = year_match.group(1)
        
        # Extract title (often between quotes or after first period)
        title_match = re.search(r'["""]([^"""]+)["""]', content)
        if title_match:
            ref['title'] = title_match.group(1)
        else:
            # Try to find title after authors
            title_match = re.search(r'(?:et\s+al\.)?\s*\.?\s*([^.]+(?:\.[^.]+)?)\.', content)
            if title_match:
                potential_title = title_match.group(1).strip()
                if len(potential_title) > 10 and len(potential_title) < 500:
                    ref['title'] = potential_title
        
        # Extract authors (before year)
        if year_match:
            year_pos = year_match.start()
            before_year = content[:year_pos]
            ref['authors'] = before_year.strip()
        
        # Extract journal/conference
        journal_match = re.search(
            r'(?:in\s+)?([A-Z][a-zA-Z\s]+(?:Journal|Conference|Proceedings|Transactions| letters)[\s\S]{10,100})',
            content
        )
        if journal_match:
            ref['journal'] = journal_match.group(1).strip()
        
        # Extract DOIs
        ref['dois'] = self._extract_dois_from_text(content)
        
        # Extract arXiv IDs
        ref['arxiv_ids'] = self._extract_arxiv_ids_from_text(content)
        
        # Extract URLs
        ref['urls'] = self._extract_urls_from_text(content)
        
        return ref
    
    def _extract_dois_from_text(self, text: str) -> List[str]:
        """
        Extract DOIs from text.
        """
        import re
        
        dois = []
        
        # DOI patterns
        doi_patterns = [
            r'\b10\.\d{4,}/[^\s]+',
            r'doi:\s*10\.\d{4,}/[^\s]+',
            r'DOI:\s*10\.\d{4,}/[^\s]+',
        ]
        
        for pattern in doi_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dois.extend(matches)
        
        # Clean up DOIs
        cleaned_dois = []
        for doi in dois:
            # Remove trailing punctuation
            doi = re.sub(r'[.,;]+$', '', doi)
            doi = doi.strip()
            if doi and doi not in cleaned_dois:
                cleaned_dois.append(doi)
        
        return cleaned_dois
    
    def _extract_arxiv_ids_from_text(self, text: str) -> List[str]:
        """
        Extract arXiv IDs from text.
        """
        import re
        
        arxiv_ids = []
        
        # arXiv ID patterns
        arxiv_patterns = [
            r'\barxiv:\s*(\d{4}\.\d{4,5}(?:\.v\d+)?)',
            r'\barXiv:\s*(\d{4}\.\d{4,5}(?:\.v\d+)?)',
            r'\b(\d{4}\.\d{4,5}(?:\.v\d+)?)\s*\[',
            r'\b(\d{4}\.\d{4,5}(?:\.v\d+)?)\s*$',
        ]
        
        for pattern in arxiv_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                full_id = f"arXiv:{match}"
                if full_id not in arxiv_ids:
                    arxiv_ids.append(full_id)
        
        return arxiv_ids
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text.
        """
        import re
        
        urls = []
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"""\'()]+'
        urls = re.findall(url_pattern, text)
        
        # Filter out DOIs (they're not URLs)
        filtered_urls = []
        for url in urls:
            if not url.startswith('http://dx.doi.org/10.'):
                url = url.rstrip('.,;')
                if url not in filtered_urls:
                    filtered_urls.append(url)
        
        return filtered_urls
    
    def _extract_identifiers_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all types of identifiers from text.
        
        Returns:
            Dict with 'dois', 'arxiv_ids', 'urls', 'emails'
        """
        import re
        
        identifiers = {
            'dois': [],
            'arxiv_ids': [],
            'urls': [],
            'emails': []
        }
        
        # Extract DOIs
        identifiers['dois'] = self._extract_dois_from_text(text)
        
        # Extract arXiv IDs
        identifiers['arxiv_ids'] = self._extract_arxiv_ids_from_text(text)
        
        # Extract URLs
        identifiers['urls'] = self._extract_urls_from_text(text)
        
        # Extract emails
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        identifiers['emails'] = re.findall(email_pattern, text)
        
        # Remove duplicates
        for key in identifiers:
            identifiers[key] = list(set(identifiers[key]))
        
        return identifiers
    
    def _format_references_markdown(self, references: List[Dict]) -> str:
        """
        Format parsed references as Markdown.
        """
        if not references:
            return ''
        
        md_lines = ['\n## References\n']
        
        for i, ref in enumerate(references, 1):
            citation = ref.get('citation', f'[{i}]')
            
            # Format: [1] Author (Year) Title. Journal.
            parts = []
            
            if ref.get('authors'):
                parts.append(ref['authors'])
            if ref.get('year'):
                parts.append(f"({ref['year']})")
            if ref.get('title'):
                parts.append(ref['title'])
            if ref.get('journal'):
                parts.append(ref['journal'])
            
            if parts:
                md_lines.append(f"{i}. {' '.join(parts)}")
            else:
                md_lines.append(f"{i}. {ref.get('raw', citation)[:200]}")
            
            # Add DOI link if available
            dois = ref.get('dois', [])
            if dois:
                md_lines.append(f"   DOI: {dois[0]}")
            
            arxiv_ids = ref.get('arxiv_ids', [])
            if arxiv_ids:
                md_lines.append(f"   arXiv: {arxiv_ids[0]}")
            
            md_lines.append('')  # Empty line between refs
        
        return '\n'.join(md_lines)
    
    def _format_references_bibliography(self, references: List[Dict]) -> str:
        """
        Format references as a bibliography entry.
        """
        if not references:
            return ''
        
        bib_entries = []
        
        for ref in references:
            if ref.get('format') == 'bibtex':
                # Already in BibTeX format
                bib_entries.append(ref.get('raw', ''))
            else:
                # Convert to BibTeX
                entry_type = ref.get('type', 'misc')
                key = self._generate_bibtex_key(ref)
                
                bibtex = [f'@{entry_type}{{{key},']
                
                if ref.get('authors'):
                    bibtex.append(f"  author = {{{ref['authors']}}},")
                if ref.get('title'):
                    bibtex.append(f"  title = {{{ref['title']}}},")
                if ref.get('year'):
                    bibtex.append(f"  year = {{{ref['year']}}},")
                if ref.get('journal'):
                    bibtex.append(f"  journal = {{{ref['journal']}}},")
                
                dois = ref.get('dois', [])
                if dois:
                    bibtex.append(f"  doi = {{{dois[0]}}},")
                
                arxiv_ids = ref.get('arxiv_ids', [])
                if arxiv_ids:
                    arxiv_id = arxiv_ids[0].replace('arXiv:', '')
                    bibtex.append(f"  eprint = {{{arxiv_id}}},")
                
                bibtex.append('}')
                bib_entries.append('\n'.join(bibtex))
        
        return '\n\n'.join(bib_entries)
    
    def _generate_bibtex_key(self, ref: Dict) -> str:
        """
        Generate a BibTeX citation key from reference.
        """
        import re
        
        # Extract author surname
        authors = ref.get('authors', '')
        if 'et al' in authors.lower():
            authors = authors.split('et al')[0]
        
        # Get first author surname
        surname = re.split(r'[,;\s]+', authors.strip())[0]
        surname = re.sub(r'[^a-zA-Z]', '', surname)
        surname = surname.lower()[:8]
        
        # Get year
        year = ref.get('year', 'xxxx')
        
        return f"{surname}{year}"
    
    # ============================================================
    # Cross-Reference Resolution
    # ============================================================
    
    def _resolve_cross_references(self, text: str, structure: Dict = None) -> Dict:
        """
        Resolve cross-references between text and document elements.
        
        Creates links between:
        - Figure references and figure captions
        - Table references and table captions
        - Equation references and equation definitions
        - Section references and section headers
        
        Args:
            text: Full document text
            structure: Document structure (sections, figures, tables, etc.)
            
        Returns:
            Dict with resolved text and reference mappings
        """
        import re
        
        if not text:
            return {
                'text': text,
                'figure_refs': [],
                'table_refs': [],
                'equation_refs': [],
                'section_refs': [],
                'element_map': {}
            }
        
        result = {
            'text': text,
            'figure_refs': [],
            'table_refs': [],
            'equation_refs': [],
            'algorithm_refs': [],
            'section_refs': [],
            'element_map': {}
        }

        # Get structure if not provided
        if structure is None:
            structure = self._extract_document_structure(text)

        # Resolve each type of reference
        result['figure_refs'] = self._resolve_figure_references(text, structure)
        result['table_refs'] = self._resolve_table_references(text, structure)
        result['equation_refs'] = self._resolve_equation_references(text, structure)
        result['algorithm_refs'] = self._resolve_algorithm_references(text, structure)
        result['section_refs'] = self._resolve_section_references(text, structure)

        # Build element map for quick lookup
        result['element_map'] = self._build_element_map(result)

        # Apply hyperlinks to text
        result['text'] = self._apply_reference_links(result)

        return result
    
    def _resolve_figure_references(self, text: str, structure: Dict) -> List[Dict]:
        """
        Resolve figure references to their captions and locations.
        
        Detects patterns like:
        - "Figure 1", "Fig. 1", "Fig 1"
        - "Figure 1.1", "Fig. 1.1" (subfigures)
        - "[1]" if it refers to a figure
        """
        import re
        
        figure_refs = []
        
        # Figure reference patterns
        fig_patterns = [
            # Standard figure references
            r'(?:[Ff]ig(?:ure)?\.?\s*)(\d+(?:\.\d+)?)',
            r'(?:[Ff]ig(?:ure)?\s+(\d+(?:\.\d+)?))',
            
            # Subfigure references
            r'(?:[Ff]ig(?:ure)?\s*)(\d+(?:\.\d+)?[a-z]?)',
            
            # Parenthesized references
            r'\((?:[Ff]ig(?:ure)?\.?\s*)(\d+(?:\.\d+)?)\)',
            
            # Bracket references
            r'\[(?:[Ff]ig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\]',
        ]
        
        figures = structure.get('figures', [])
        
        for pattern in fig_patterns:
            for match in re.finditer(pattern, text):
                fig_num = match.group(1)
                
                # Find figure caption
                caption_info = self._find_figure_caption_info(structure, fig_num)
                
                # Check if reference is valid
                is_valid = self._is_valid_figure_ref(fig_num, figures)
                
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                
                ref = {
                    'ref_text': match.group(0),
                    'number': fig_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'has_caption': caption_info is not None,
                    'caption': caption_info.get('text', '') if caption_info else '',
                    'page': caption_info.get('page') if caption_info else None,
                    'is_valid': is_valid
                }
                
                # Check if already added
                if not any(r['number'] == fig_num and r['position'] == match.start() for r in figure_refs):
                    figure_refs.append(ref)
        
        return figure_refs
    
    def _find_figure_caption_info(self, structure: Dict, fig_num: str) -> Optional[Dict]:
        """Find figure caption for a given figure number."""
        # Search in abstract and sections for figure captions
        abstract = structure.get('abstract', {})
        abstract_text = abstract.get('abstract', '')
        
        # Pattern: "Figure X" or "Fig. X" followed by caption text
        caption_patterns = [
            rf'(?:[Ff]ig(?:ure)?\.?\s*{fig_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
            rf'(?:[Ff]ig(?:ure)?\s+{fig_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
        ]
        
        for pattern in caption_patterns:
            match = re.search(pattern, abstract_text + ' ' + ' '.join(
                s.get('content', '') for s in structure.get('sections', [])
            ))
            if match:
                return {
                    'text': match.group(1).strip()[:200],
                    'page': 'abstract'
                }
        
        # Check figure list in structure
        for fig in structure.get('figures', []):
            if fig.get('number') == fig_num:
                return {
                    'text': fig.get('caption', ''),
                    'page': fig.get('page', 'unknown')
                }
        
        return None
    
    def _is_valid_figure_ref(self, fig_num: str, figures: List[Dict]) -> bool:
        """Check if a figure reference is valid."""
        # If we have a figure list, validate against it
        if figures:
            for fig in figures:
                if fig.get('number') == fig_num:
                    return True
            return False  # Reference but no matching figure
        
        # No figure list - assume valid reference
        return True
    
    def _resolve_table_references(self, text: str, structure: Dict) -> List[Dict]:
        """
        Resolve table references to their captions.
        
        Detects patterns like:
        - "Table 1", "Tab. 1", "Tab 1"
        - "Table 1.1" (subtables)
        """
        import re
        
        table_refs = []
        
        # Table reference patterns
        table_patterns = [
            r'(?:[Tt]ab(?:le)?\.?\s*)(\d+(?:\.\d+)?)',
            r'(?:[Tt]ab(?:le)?\s+(\d+(?:\.\d+)?))',
            r'\((?:[Tt]ab(?:le)?\.?\s*)(\d+(?:\.\d+)?)\)',
            r'\[(?:[Tt]ab(?:le)?\.?\s*(\d+(?:\.\d+)?)\]',
        ]
        
        tables = structure.get('tables', [])
        
        for pattern in table_patterns:
            for match in re.finditer(pattern, text):
                tab_num = match.group(1)
                
                # Find table caption
                caption_info = self._find_table_caption_info(structure, tab_num)
                
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                
                ref = {
                    'ref_text': match.group(0),
                    'number': tab_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'has_caption': caption_info is not None,
                    'caption': caption_info.get('text', '') if caption_info else '',
                    'page': caption_info.get('page') if caption_info else None
                }
                
                if not any(r['number'] == tab_num and r['position'] == match.start() for r in table_refs):
                    table_refs.append(ref)
        
        return table_refs
    
    def _find_table_caption_info(self, structure: Dict, tab_num: str) -> Optional[Dict]:
        """Find table caption for a given table number."""
        # Search in sections for table captions
        sections = structure.get('sections', [])
        
        for section in sections:
            content = section.get('content', '')
            
            caption_patterns = [
                rf'(?:[Tt]ab(?:le)?\.?\s*{tab_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
                rf'(?:[Tt]ab(?:le)?\s+{tab_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
            ]
            
            for pattern in caption_patterns:
                match = re.search(pattern, content)
                if match:
                    return {
                        'text': match.group(1).strip()[:200],
                        'page': section.get('number', section.get('title', 'unknown'))
                    }
        
        # Check table list
        for tab in structure.get('tables', []):
            if tab.get('number') == tab_num:
                return {
                    'text': tab.get('caption', ''),
                    'page': tab.get('page', 'unknown')
                }
        
        return None
    
    def _resolve_algorithm_references(self, text: str, structure: Dict) -> List[Dict]:
        """
        Resolve algorithm references to their definitions.

        Detects patterns like:
        - "Algorithm 1", "Alg. 1"
        - "Procedure 1"
        - "Listing 1", "Code 1"
        """
        import re

        algorithm_refs = []

        # Algorithm reference patterns
        algo_patterns = [
            r'(?:[Aa]lg(?:orithm)?\.?\s*)(\d+(?:\.\d+)?)',
            r'(?:[Aa]lg(?:orithm)?\s+(\d+(?:\.\d+)?))',
            r'(?:[Pp]rocedure\.?\s*)(\d+(?:\.\d+)?)',
            r'(?:[Ll]isting\.?\s*)(\d+(?:\.\d+)?)',
            r'(?:[Cc]ode\s+(?:fragment\s*)?)(\d+(?:\.\d+)?)',
        ]

        algorithms = structure.get('algorithms', [])

        for pattern in algo_patterns:
            for match in re.finditer(pattern, text):
                algo_num = match.group(1)

                # Find algorithm caption/definition
                caption_info = self._find_algorithm_caption(structure, algo_num)

                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]

                ref = {
                    'ref_text': match.group(0),
                    'number': algo_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'has_caption': caption_info is not None,
                    'caption': caption_info.get('text', '') if caption_info else '',
                    'page': caption_info.get('page') if caption_info else None
                }

                if not any(r['number'] == algo_num and r['position'] == match.start() for r in algorithm_refs):
                    algorithm_refs.append(ref)

        return algorithm_refs

    def _find_algorithm_caption(self, structure: Dict, algo_num: str) -> Optional[Dict]:
        """Find algorithm caption for a given number."""
        import re

        sections = structure.get('sections', [])

        for section in sections:
            content = section.get('content', '')

            caption_patterns = [
                rf'(?:[Aa]lg(?:orithm)?\.?\s*{algo_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
                rf'(?:[Aa]lg(?:orithm)?\s+{algo_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
                rf'(?:[Pp]rocedure\.?\s*{algo_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
                rf'(?:[Ll]isting\.?\s*{algo_num}[:\.]?\s*([^\n.]+(?:\.[^\n.]+)?)',
            ]

            for pattern in caption_patterns:
                match = re.search(pattern, content)
                if match:
                    return {
                        'text': match.group(1).strip()[:200],
                        'page': section.get('number', section.get('title', 'unknown'))
                    }

        # Check algorithm list from structure (extracted by _extract_algorithms)
        for algo in structure.get('algorithms', []):
            algo_num_str = str(algo_num)

            # Match by number or by title containing the number
            if algo.get('number') == algo_num_str:
                return {
                    'text': algo.get('title', algo.get('caption', '')),
                    'page': algo.get('page', 'unknown'),
                    'input': algo.get('input', []),
                    'output': algo.get('output', []),
                    'keywords': algo.get('keywords', [])
                }

            # Also check if title contains the algorithm number
            title = algo.get('title', '').lower()
            if algo_num_str in title:
                return {
                    'text': algo.get('title', algo.get('caption', '')),
                    'page': algo.get('page', 'unknown'),
                    'input': algo.get('input', []),
                    'output': algo.get('output', []),
                    'keywords': algo.get('keywords', [])
                }

        return None

    def _resolve_equation_references(self, text: str, structure: Dict) -> List[Dict]:
        """
        Resolve equation references to their definitions.

        Detects patterns like:
        - "Equation (1)", "Eq. (1)", "Eq (1)"
        - "(1)" in mathematical context
        - "Equation 1" without parentheses
        """
        import re

        equation_refs = []

        # Equation reference patterns
        eq_patterns = [
            r'(?:[Ee]q(?:uation)?\.?\s*)(\(\d+(?:\.\d+)?\))',
            r'(?:[Ee]q(?:uation)?\s+(\d+(?:\.\d+)?)',
            r'\(\s*(\d+(?:\.\d+)?)\s*\)',  # (1) or (1.2)
        ]
        
        # Mathematical context indicators
        math_context = [
            'where', 'from', 'in', 'as shown in', 'see',
            'substituting', 'rearranging', 'expanding'
        ]
        
        equations = structure.get('equations', [])
        
        for pattern in eq_patterns:
            for match in re.finditer(pattern, text):
                eq_num = match.group(1)
                
                # Get context to verify it's an equation reference
                context_start = max(0, match.start() - 50)
                context = text[context_start:match.start()]
                
                # Check for math context
                has_math_context = any(
                    kw in context.lower() for kw in math_context
                )
                
                # Find equation definition
                eq_def = self._find_equation_definition(structure, eq_num)
                
                ref = {
                    'ref_text': match.group(0),
                    'number': eq_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'has_definition': eq_def is not None,
                    'definition': eq_def.get('text', '')[:100] if eq_def else '',
                    'page': eq_def.get('page') if eq_def else None,
                    'has_math_context': has_math_context
                }
                
                if not any(r['number'] == eq_num and r['position'] == match.start() for r in equation_refs):
                    equation_refs.append(ref)
        
        return equation_refs
    
    def _find_equation_definition(self, structure: Dict, eq_num: str) -> Optional[Dict]:
        """Find equation definition for a given equation number."""
        # Search in sections for equation definitions
        sections = structure.get('sections', [])
        
        for section in sections:
            content = section.get('content', '')
            
            # Look for standalone equation numbers
            eq_pattern = rf'(?:^|\n)\s*\(\s*{eq_num}\s*\)\s*[:\s]+([^\n]+)'
            match = re.search(eq_pattern, content)
            if match:
                return {
                    'text': match.group(1).strip()[:200],
                    'page': section.get('number', section.get('title', 'unknown'))
                }
            
            # Look for displayed equations
            display_pattern = r'\$\$?.*?\{?\s*' + re.escape(eq_num) + r'\s*\}?.*?\$?\$?'
            if re.search(display_pattern, content):
                return {
                    'text': f'Equation {eq_num}',
                    'page': section.get('number', section.get('title', 'unknown'))
                }
        
        return None
    
    def _resolve_section_references(self, text: str, structure: Dict) -> List[Dict]:
        """
        Resolve section references to their headers.
        
        Detects patterns like:
        - "Section 1", "Sec. 1"
        - "In Section 2"
        - "As discussed in Section 3.1"
        """
        import re
        
        section_refs = []
        
        sections = structure.get('sections', [])
        
        # Section reference patterns
        section_patterns = [
            r'(?:[Ss]ection|Sec\.?)\s+(\d+(?:\.\d+)?)',
            r'(?:[Ss]ection|Sec\.?)\s+([A-Z])',  # Roman/Letter sections
            r'(?:chapter|Chapter)\s+(\d+(?:\.\d+)?)',
            r'(?:[Pp]art)\s+([A-Z])',
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                sec_num = match.group(1)
                
                # Find section header
                header = self._find_section_header(sections, sec_num)
                
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                
                ref = {
                    'ref_text': match.group(0),
                    'number': sec_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'has_header': header is not None,
                    'header': header.get('title', '') if header else '',
                    'page': header.get('number', header.get('title', 'unknown')) if header else None
                }
                
                if not any(r['number'] == sec_num and r['position'] == match.start() for r in section_refs):
                    section_refs.append(ref)
        
        return section_refs
    
    def _find_section_header(self, sections: List[Dict], sec_num: str) -> Optional[Dict]:
        """Find section header for a given section number."""
        for section in sections:
            if section.get('number') == sec_num:
                return section
            
            # Also check title for numbered sections
            title = section.get('title', '')
            if re.search(rf'^{re.escape(sec_num)}\.?\s', title):
                return section
        
        return None
    
    def _build_element_map(self, result: Dict) -> Dict:
        """
        Build a map of all referenceable elements for quick lookup.
        """
        element_map = {
            'figures': {},
            'tables': {},
            'equations': {},
            'algorithms': {},
            'sections': {}
        }

        # Map figures
        for fig in result.get('figure_refs', []):
            if fig.get('has_caption') and fig.get('number'):
                element_map['figures'][fig['number']] = {
                    'caption': fig.get('caption', ''),
                    'page': fig.get('page'),
                    'position': fig.get('position')
                }

        # Map tables
        for tab in result.get('table_refs', []):
            if tab.get('has_caption') and tab.get('number'):
                element_map['tables'][tab['number']] = {
                    'caption': tab.get('caption', ''),
                    'page': tab.get('page'),
                    'position': tab.get('position')
                }

        # Map equations
        for eq in result.get('equation_refs', []):
            if eq.get('has_definition') and eq.get('number'):
                element_map['equations'][eq['number']] = {
                    'definition': eq.get('definition', ''),
                    'page': eq.get('page'),
                    'position': eq.get('position')
                }

        # Map algorithms
        for algo in result.get('algorithm_refs', []):
            if algo.get('has_caption') and algo.get('number'):
                element_map['algorithms'][algo['number']] = {
                    'caption': algo.get('caption', ''),
                    'page': algo.get('page'),
                    'position': algo.get('position')
                }

        # Map sections
        for sec in result.get('section_refs', []):
            if sec.get('has_header') and sec.get('number'):
                element_map['sections'][sec['number']] = {
                    'header': sec.get('header', ''),
                    'page': sec.get('page'),
                    'position': sec.get('position')
                }

        return element_map

    def _apply_reference_links(self, result: Dict) -> str:
        """
        Apply hyperlink-style markup to references in text.
        
        Converts:
        - "Figure 1" → "**Figure 1**{caption at page X}"
        - "Table 2" → "**Table 2**{caption}"
        - "Equation (1)" → "**Equation (1)**{definition}"
        """
        text = result['text']
        
        # Apply figure links
        for fig in result.get('figure_refs', []):
            if fig.get('has_caption'):
                link = f"**{fig['ref_text']}**"
                page_info = f" [page {fig['page']}]" if fig.get('page') else ""
                tooltip = f"{{{fig['caption'][:50]}...}}"
                replacement = f"{link}{page_info}{tooltip}"
            else:
                link = f"**{fig['ref_text']}**"
                replacement = f"{link}"
            
            text = text.replace(fig['ref_text'], replacement, 1)
        
        # Apply table links
        for tab in result.get('table_refs', []):
            if tab.get('has_caption'):
                link = f"**{tab['ref_text']}**"
                page_info = f" [page {tab['page']}]" if tab.get('page') else ""
                tooltip = f"{{{tab['caption'][:50]}...}}"
                replacement = f"{link}{page_info}{tooltip}"
            else:
                link = f"**{tab['ref_text']}**"
                replacement = f"{link}"
            
            text = text.replace(tab['ref_text'], replacement, 1)
        
        # Apply equation links
        for eq in result.get('equation_refs', []):
            if eq.get('has_definition'):
                link = f"**{eq['ref_text']}**"
                definition = eq.get('definition', '')[:30]
                tooltip = f"{{{definition}...}}"
                replacement = f"{link}{tooltip}"
            else:
                link = f"**{eq['ref_text']}**"
                replacement = f"{link}"
            
            text = text.replace(eq['ref_text'], replacement, 1)

        # Apply algorithm links
        for algo in result.get('algorithm_refs', []):
            if algo.get('has_caption'):
                link = f"**{algo['ref_text']}**"
                page_info = f" [page {algo['page']}]" if algo.get('page') else ""
                tooltip = f"{{{algo['caption'][:50]}...}}"
                replacement = f"{link}{page_info}{tooltip}"
            else:
                link = f"**{algo['ref_text']}**"
                replacement = f"{link}"

            text = text.replace(algo['ref_text'], replacement, 1)

        return text

    def _get_reference_summary(self, result: Dict) -> Dict:
        """
        Generate a comprehensive summary of all cross-references.

        Returns:
            Dict with summary statistics and categorized references
        """
        summary = {
            'total_references': 0,
            'by_type': {
                'figures': {'total': 0, 'with_captions': 0, 'without_captions': 0},
                'tables': {'total': 0, 'with_captions': 0, 'without_captions': 0},
                'equations': {'total': 0, 'with_definitions': 0, 'without_definitions': 0},
                'algorithms': {'total': 0, 'with_captions': 0, 'without_captions': 0},
                'sections': {'total': 0, 'with_headers': 0, 'without_headers': 0}
            },
            'unresolved_refs': [],
            'reference_density': 0.0
        }

        # Count figures
        for fig in result.get('figure_refs', []):
            summary['total_references'] += 1
            summary['by_type']['figures']['total'] += 1
            if fig.get('has_caption'):
                summary['by_type']['figures']['with_captions'] += 1
            else:
                summary['by_type']['figures']['without_captions'] += 1
                summary['unresolved_refs'].append({
                    'type': 'figure',
                    'ref': fig.get('ref_text'),
                    'number': fig.get('number')
                })

        # Count tables
        for tab in result.get('table_refs', []):
            summary['total_references'] += 1
            summary['by_type']['tables']['total'] += 1
            if tab.get('has_caption'):
                summary['by_type']['tables']['with_captions'] += 1
            else:
                summary['by_type']['tables']['without_captions'] += 1
                summary['unresolved_refs'].append({
                    'type': 'table',
                    'ref': tab.get('ref_text'),
                    'number': tab.get('number')
                })

        # Count equations
        for eq in result.get('equation_refs', []):
            summary['total_references'] += 1
            summary['by_type']['equations']['total'] += 1
            if eq.get('has_definition'):
                summary['by_type']['equations']['with_definitions'] += 1
            else:
                summary['by_type']['equations']['without_definitions'] += 1
                summary['unresolved_refs'].append({
                    'type': 'equation',
                    'ref': eq.get('ref_text'),
                    'number': eq.get('number')
                })

        # Count algorithms
        for algo in result.get('algorithm_refs', []):
            summary['total_references'] += 1
            summary['by_type']['algorithms']['total'] += 1
            if algo.get('has_caption'):
                summary['by_type']['algorithms']['with_captions'] += 1
            else:
                summary['by_type']['algorithms']['without_captions'] += 1
                summary['unresolved_refs'].append({
                    'type': 'algorithm',
                    'ref': algo.get('ref_text'),
                    'number': algo.get('number')
                })

        # Count sections
        for sec in result.get('section_refs', []):
            summary['total_references'] += 1
            summary['by_type']['sections']['total'] += 1
            if sec.get('has_header'):
                summary['by_type']['sections']['with_headers'] += 1
            else:
                summary['by_type']['sections']['without_headers'] += 1

        # Calculate reference density (refs per 1000 chars)
        text = result.get('text', '')
        if len(text) > 0:
            summary['reference_density'] = round(
                summary['total_references'] / (len(text) / 1000), 2
            )

        return summary

    def _format_references_for_markdown(self, result: Dict) -> str:
        """
        Format cross-references as a markdown reference section.

        Creates a clean, organized list of all references with their
        captions/definitions for easy navigation.
        """
        lines = []
        lines.append('\n## Reference Index\n')

        # Figures section
        if result.get('figure_refs'):
            lines.append('### Figures\n')
            seen_figs = set()
            for fig in result['figure_refs']:
                num = fig.get('number')
                if num not in seen_figs:
                    seen_figs.add(num)
                    caption = fig.get('caption', 'No caption found')
                    page = fig.get('page', 'Unknown')
                    lines.append(f'- **Figure {num}** (page {page}): {caption[:100]}')

        # Tables section
        if result.get('table_refs'):
            lines.append('\n### Tables\n')
            seen_tabs = set()
            for tab in result['table_refs']:
                num = tab.get('number')
                if num not in seen_tabs:
                    seen_tabs.add(num)
                    caption = tab.get('caption', 'No caption found')
                    page = tab.get('page', 'Unknown')
                    lines.append(f'- **Table {num}** (page {page}): {caption[:100]}')

        # Equations section
        if result.get('equation_refs'):
            lines.append('\n### Equations\n')
            seen_eqs = set()
            for eq in result['equation_refs']:
                num = eq.get('number')
                if num not in seen_eqs:
                    seen_eqs.add(num)
                    definition = eq.get('definition', 'No definition found')
                    page = eq.get('page', 'Unknown')
                    lines.append(f'- **Equation {num}** (page {page}): {definition[:100]}')

        # Algorithms section
        if result.get('algorithm_refs'):
            lines.append('\n### Algorithms\n')
            seen_algos = set()
            for algo in result['algorithm_refs']:
                num = algo.get('number')
                if num not in seen_algos:
                    seen_algos.add(num)
                    caption = algo.get('caption', 'No caption found')
                    page = algo.get('page', 'Unknown')
                    lines.append(f'- **Algorithm {num}** (page {page}): {caption[:100]}')

        # Sections section
        if result.get('section_refs'):
            lines.append('\n### Sections\n')
            seen_secs = set()
            for sec in result['section_refs']:
                num = sec.get('number')
                if num not in seen_secs:
                    seen_secs.add(num)
                    header = sec.get('header', 'No header found')
                    lines.append(f'- **Section {num}**: {header[:80]}')

        return '\n'.join(lines)

    def _apply_enhanced_links(self, result: Dict, style: str = 'tooltip') -> str:
        """
        Apply enhanced hyperlink-style markup with different styles.

        Args:
            result: Cross-reference resolution result
            style: Link style - 'tooltip', 'inline', 'numbered', 'markdown'

        Returns:
            Text with enhanced reference links applied
        """
        import re

        text = result['text']

        if style == 'tooltip':
            return self._apply_reference_links(result)

        elif style == 'inline':
            # Inline references: Figure 1 → [Figure 1: caption]
            for fig in result.get('figure_refs', []):
                if fig.get('has_caption'):
                    link = f"[{fig['ref_text']}: {fig['caption'][:30]}...]"
                else:
                    link = f"[{fig['ref_text']}]"
                text = text.replace(fig['ref_text'], link, 1)

            for tab in result.get('table_refs', []):
                if tab.get('has_caption'):
                    link = f"[{tab['ref_text']}: {tab['caption'][:30]}...]"
                else:
                    link = f"[{tab['ref_text']}]"
                text = text.replace(tab['ref_text'], link, 1)

            for eq in result.get('equation_refs', []):
                if eq.get('has_definition'):
                    link = f"[{eq['ref_text']}: {eq['definition'][:30]}...]"
                else:
                    link = f"[{eq['ref_text']}]"
                text = text.replace(eq['ref_text'], link, 1)

            for algo in result.get('algorithm_refs', []):
                if algo.get('has_caption'):
                    link = f"[{algo['ref_text']}: {algo['caption'][:30]}...]"
                else:
                    link = f"[{algo['ref_text']}]"
                text = text.replace(algo['ref_text'], link, 1)

        elif style == 'numbered':
            # Numbered superscript style: Figure 1 → [1]
            ref_counter = {'fig': 1, 'tab': 1, 'eq': 1, 'algo': 1}
            fig_map = {}
            tab_map = {}
            eq_map = {}
            algo_map = {}

            for fig in result.get('figure_refs', []):
                num = fig.get('number')
                if num not in fig_map:
                    fig_map[num] = ref_counter['fig']
                    ref_counter['fig'] += 1
                text = re.sub(
                    re.escape(fig['ref_text']),
                    f"[{fig_map[num]}]",
                    text,
                    count=1
                )

            for tab in result.get('table_refs', []):
                num = tab.get('number')
                if num not in tab_map:
                    tab_map[num] = ref_counter['tab']
                    ref_counter['tab'] += 1
                text = re.sub(
                    re.escape(tab['ref_text']),
                    f"[{tab_map[num]}]",
                    text,
                    count=1
                )

            for eq in result.get('equation_refs', []):
                num = eq.get('number')
                if num not in eq_map:
                    eq_map[num] = ref_counter['eq']
                    ref_counter['eq'] += 1
                text = re.sub(
                    re.escape(eq['ref_text']),
                    f"[{eq_map[num]}]",
                    text,
                    count=1
                )

            for algo in result.get('algorithm_refs', []):
                num = algo.get('number')
                if num not in algo_map:
                    algo_map[num] = ref_counter['algo']
                    ref_counter['algo'] += 1
                text = re.sub(
                    re.escape(algo['ref_text']),
                    f"[{algo_map[num]}]",
                    text,
                    count=1
                )

        elif style == 'markdown':
            # Standard markdown links: Figure 1 → [Figure 1](#fig-1)
            for fig in result.get('figure_refs', []):
                link = f"[{fig['ref_text']}](#fig-{fig['number']})"
                text = text.replace(fig['ref_text'], link, 1)

            for tab in result.get('table_refs', []):
                link = f"[{tab['ref_text']}](#tab-{tab['number']})"
                text = text.replace(tab['ref_text'], link, 1)

            for eq in result.get('equation_refs', []):
                link = f"[{eq['ref_text']}](#eq-{eq['number']})"
                text = text.replace(eq['ref_text'], link, 1)

            for algo in result.get('algorithm_refs', []):
                link = f"[{algo['ref_text']}](#algo-{algo['number']})"
                text = text.replace(algo['ref_text'], link, 1)

            for sec in result.get('section_refs', []):
                link = f"[{sec['ref_text']}](#sec-{sec['number']})"
                text = text.replace(sec['ref_text'], link, 1)

        return text

    def _format_references_html(self, result: Dict) -> str:
        """
        Format cross-references as HTML for web display.
        
        Creates clickable links to figures, tables, equations, and sections.
        """
        html_lines = ['\n## Cross-References\n']
        
        # Figures section
        if result.get('figure_refs'):
            html_lines.append('<h3>Figures</h3>')
            html_lines.append('<ul class="figure-list">')
            for fig in result['figure_refs']:
                if fig.get('has_caption'):
                    html_lines.append(
                        f'<li><a href="#fig-{fig["number"]}">'
                        f'Figure {fig["number"]}</a>: {fig["caption"][:100]}...</li>'
                    )
            html_lines.append('</ul>')
        
        # Tables section
        if result.get('table_refs'):
            html_lines.append('<h3>Tables</h3>')
            html_lines.append('<ul class="table-list">')
            for tab in result['table_refs']:
                if tab.get('has_caption'):
                    html_lines.append(
                        f'<li><a href="#tab-{tab["number"]}">'
                        f'Table {tab["number"]}</a>: {tab["caption"][:100]}...</li>'
                    )
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)

    # ============================================================
    # Footnote and Endnote Handling
    # ============================================================

    def _extract_footnotes(self, text: str) -> Dict:
        """
        Extract and process footnotes/endnotes from document text.

        This method handles:
        - Footnotes (page-bottom references)
        - Endnotes (document-end references)
        - Various footnote marker styles: [1], (1), superscripts, asterisks
        - Inline references that look like footnotes

        Returns:
            Dict with:
                - 'cleaned_text': Main text with footnote markers removed
                - 'footnotes': List of extracted footnotes
                - 'endnotes': List of extracted endnotes
                - 'superscript_refs': Mapping of reference numbers to content
                - 'footnote_positions': Map of footnote positions in text
        """
        import re

        if not text:
            return {
                'cleaned_text': '',
                'footnotes': [],
                'endnotes': [],
                'superscript_refs': {},
                'footnote_positions': {}
            }

        result = {
            'cleaned_text': '',
            'footnotes': [],
            'endnotes': [],
            'superscript_refs': {},
            'footnote_positions': {}
        }

        # Find footnote and endnote sections
        footnote_section = self._find_footnote_section(text)
        endnote_section = self._find_endnote_section(text)

        # Parse footnote definitions from section
        if footnote_section:
            result['footnotes'] = self._parse_footnote_definitions(footnote_section, is_endnote=False)

        # Parse endnote definitions
        if endnote_section:
            result['endnotes'] = self._parse_footnote_definitions(endnote_section, is_endnote=True)

        # Extract inline footnote references from main text
        inline_footnotes = self._extract_inline_footnotes(text)
        result['footnotes'].extend(inline_footnotes)

        # Build superscript reference mapping
        all_notes = result['footnotes'] + result['endnotes']
        for note in all_notes:
            if 'number' in note:
                result['superscript_refs'][str(note['number'])] = note.get('content', '')

        # Remove footnote markers from main text
        cleaned_text = self._remove_footnote_markers(text)
        result['cleaned_text'] = cleaned_text

        # Track footnote positions
        result['footnote_positions'] = self._map_footnote_positions(text, result['footnotes'])

        return result

    def _find_footnote_section(self, text: str) -> Optional[str]:
        """
        Find the footnote section in document text.

        Looks for common footnote section markers and headers.

        Returns:
            Text content of footnote section, or None if not found
        """
        import re

        if not text:
            return None

        # Try to find the position where footnotes begin
        footnote_header_patterns = [
            # Standard footnote headers
            r'(?:\n|^)\s*[Aa]cknowledgment\s*[Ss]ections?\s*(?:\n|$)',
            r'(?:\n|^)\s*[Ff]ootnotes?\s*(?:\n|$)',
            r'(?:\n|^)\s*[Nn]otes?\s*(?:\n|$)',
            r'(?:\n|^)\s*--+\s*[Ff]ootnotes?\s*--+',
            r'(?:\n|^)\s*\*\*+\s*[Ff]ootnotes?\s*\*\*+',

            # Numbered section that looks like footnotes (e.g., "1. Some note...")
            r'(?:\n|^)\s*\d+\.\s+[A-Z][a-z]+.*?(?=\n\s*\d+\.|\n\s*[A-Z][a-z]+\s+\d|\n\s*References|\n\s*Bibliography)',

            # ArXiv style footnotes at bottom of page (look for small text sections)
            r'(?:\n|^)\s*[†‡*§¶]\s+\S.*?(?=\n\n|\n\s*[A-Z][a-z]+\s+\d|\n\s*References)',
        ]

        for pattern in footnote_header_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                # Extract section content
                start_pos = match.end()

                # Try to find where footnotes end (next major section or double newline)
                end_patterns = [
                    r'\n\s*\n\s*(?:[A-Z][a-z]+\s+\d|\d+\.\s+[A-Z][a-z]+)',  # Next section
                    r'\n\s*\n\s*References',
                    r'\n\s*\n\s*Bibliography',
                    r'\n\s*\n\s*Appendices?',
                ]

                end_pos = len(text)
                for end_pat in end_patterns:
                    end_match = re.search(end_pat, text[start_pos:], re.IGNORECASE)
                    if end_match:
                        end_pos = min(end_pos, start_pos + end_match.start())

                return text[start_pos:end_pos].strip()

        return None

    def _find_endnote_section(self, text: str) -> Optional[str]:
        """
        Find the endnote section in document text.

        Endnotes are typically at the end of a document, after References.

        Returns:
            Text content of endnote section, or None if not found
        """
        import re

        if not text:
            return None

        endnote_header_patterns = [
            r'(?:\n|^)\s*[Ee]nd\s*[Nn]otes?\s*(?:\n|$)',
            r'(?:\n|^)\s*Notes?\s*(?:\n|$)(?=.*?(?:[Rr]eferences?|[Bb]ibliography))',
            r'(?:\n|^)\s*[Ss]upplementary\s*[Nn]otes?\s*(?:\n|$)',
        ]

        for pattern in endnote_header_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Endnotes go until end of text or next major section
                return text[match.end():].strip()

        return None

    def _parse_footnote_definitions(self, section_text: str, is_endnote: bool = False) -> List[Dict]:
        """
        Parse footnote/endnote definitions from section text.

        Handles various formats:
        - Numbered: "1. This is a footnote"
        - Asterisk: "* This is a footnote"
        - Superscript: "¹ This is a footnote"
        - Bracket: "[1] This is a footnote"

        Args:
            section_text: The text content of footnote/endnote section
            is_endnote: Whether this is endnotes (affects section header detection)

        Returns:
            List of footnote dictionaries with number, content, and position
        """
        import re

        footnotes = []

        if not section_text:
            return footnotes

        # Pattern for footnote definitions with various markers
        footnote_patterns = [
            # Numbered footnotes: "1. content", "1 content", "1) content"
            (r'(?:^|\n)\s*(\d+)\.?\s+([^\n]+)', 'numbered'),

            # Asterisk style: "* content", "** content", "*** content"
            (r'(?:^|\n)\s*([*†‡§¶])\s*\s*([^\n]+)', 'asterisk'),

            # Bracket style: "[1] content"
            (r'(?:^|\n)\s*\[(\d+)\]\s*([^\n]+)', 'bracket'),

            # Superscript style (Unicode): "¹ content"
            (r'(?:^|\n)\s*([\u00b9\u00b2\u00b3\u2070-\u2079])\s*([^\n]+)', 'superscript'),
        ]

        for pattern, style in footnote_patterns:
            matches = re.finditer(pattern, section_text, re.MULTILINE)

            for match in matches:
                try:
                    if style == 'asterisk':
                        # Asterisks can be multiple, use them as number
                        marker = match.group(1)
                        number = {'*': 1, '†': 2, '‡': 3, '§': 4, '¶': 5}.get(marker, 1)
                        if marker == '*':
                            # Count consecutive asterisks
                            number = len(match.group(0).strip().rstrip(marker).replace('*', '')) + 1
                    else:
                        number = int(match.group(1))

                    content = match.group(2).strip()

                    # Clean up content
                    content = re.sub(r'\s+', ' ', content)
                    content = content.rstrip('.')

                    if content:
                        footnotes.append({
                            'number': number,
                            'content': content,
                            'style': style,
                            'is_endnote': is_endnote,
                            'position': match.start()
                        })

                except (ValueError, IndexError):
                    continue

        # Sort by number and remove duplicates
        seen = set()
        unique_footnotes = []
        for footnote in sorted(footnotes, key=lambda x: x.get('number', 0)):
            key = (footnote.get('number'), footnote.get('content', '')[:50])
            if key not in seen:
                seen.add(key)
                unique_footnotes.append(footnote)

        return unique_footnotes

    def _extract_inline_footnotes(self, text: str) -> List[Dict]:
        """
        Extract inline footnote-like references from main text.

        These are references that look like footnotes but are embedded in text.

        Returns:
            List of inline footnote dictionaries
        """
        import re

        inline_footnotes = []

        # Pattern for inline references that might be footnotes
        inline_patterns = [
            # Bracket references at end of sentences: "text[1]"
            r'([^\[\]]+)\[(\d+)\](?:\.|$)',
            # Parenthesized references: "text(1)"
            r'([^(]+)\((\d+)\)(?:\.|$)',
        ]

        for pattern in inline_patterns:
            for match in re.finditer(pattern, text):
                try:
                    context = match.group(1).strip()
                    number = int(match.group(2))

                    # Check if this looks like a footnote (short context, reference at end)
                    if len(context) < 200 and context.split()[-1:]:
                        inline_footnotes.append({
                            'number': number,
                            'content': f"Reference: {context[-100:]}",
                            'style': 'inline',
                            'is_endnote': False,
                            'position': match.start()
                        })
                except (ValueError, IndexError):
                    continue

        return inline_footnotes

    def _remove_footnote_markers(self, text: str) -> str:
        """
        Remove footnote markers from text while preserving readability.

        Args:
            text: Original text with footnote markers

        Returns:
            Cleaned text with footnote markers removed
        """
        import re

        if not text:
            return text

        cleaned = text

        # Remove superscript numbers: [1], (1), ¹, ²
        markers_to_remove = [
            r'\[\d+\]',           # [1], [123]
            r'\(\d+\)',           # (1), (123)
            r'\s+\d+\s*$',        # trailing numbers on lines
            r'[\u00b9\u00b2\u00b3\u2070-\u2079]',  # Unicode superscripts
            r'\s*[*†‡§¶]\s*',     # Asterisk and symbol markers
        ]

        for pattern in markers_to_remove:
            cleaned = re.sub(pattern, '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned.strip()

    def _map_footnote_positions(self, text: str, footnotes: List[Dict]) -> Dict[int, Dict]:
        """
        Map footnote numbers to their positions in the original text.

        Args:
            text: Original text
            footnotes: List of extracted footnotes

        Returns:
            Dict mapping footnote number to position info
        """
        import re

        positions = {}

        for footnote in footnotes:
            number = footnote.get('number')
            if not number:
                continue

            # Try to find the reference in text
            patterns = [
                rf'\[{number}\]',           # [1]
                rf'\({number}\)',            # (1)
                rf'\s{number}\s',            # space number space
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    positions[number] = {
                        'start': match.start(),
                        'end': match.end(),
                        'context': text[max(0, match.start() - 50):min(len(text), match.end() + 50)].strip()
                    }
                    break

        return positions

    def _format_footnotes_for_output(self, footnote_result: Dict) -> str:
        """
        Format footnotes for inclusion in output documents.

        Args:
            footnote_result: Result from _extract_footnotes

        Returns:
            Formatted footnote text
        """
        lines = []

        if footnote_result.get('footnotes'):
            lines.append('\n## Footnotes\n')
            for footnote in footnote_result['footnotes']:
                num = footnote.get('number', '?')
                content = footnote.get('content', '')
                lines.append(f'^{num}. {content}')

        if footnote_result.get('endnotes'):
            lines.append('\n## Endnotes\n')
            for endnote in footnote_result['endnotes']:
                num = endnote.get('number', '?')
                content = endnote.get('content', '')
                lines.append(f'{num}. {content}')

        if footnote_result.get('superscript_refs'):
            lines.append('\n## Reference Mapping\n')
            for num, content in footnote_result['superscript_refs'].items():
                lines.append(f'[{num}] → {content[:100]}...' if len(content) > 100 else f'[{num}] → {content}')

        return '\n'.join(lines)

    # ============================================================
    # DOI and arXiv ID Extraction
    # ============================================================
    
    def _extract_all_identifiers(self, text: str) -> Dict:
        """
        Comprehensive identifier extraction from entire document.
        
        Extracts:
        - DOIs (Digital Object Identifiers)
        - arXiv IDs (multiple formats)
        - URLs (http/https/ftp)
        - Emails
        - ISBNs (if present)
        - ORCIDs (if present)
        
        Returns:
            Dict with comprehensive identifier information
        """
        import re
        
        result = {
            'dois': [],
            'arxiv_ids': [],
            'urls': [],
            'emails': [],
            'isbns': [],
            'orcids': [],
            'metadata': {
                'total_identifiers': 0,
                'unique_dois': 0,
                'unique_arxiv_ids': 0,
            }
        }
        
        if not text:
            return result
        
        # Extract all identifier types
        result['dois'] = self._extract_dois_comprehensive(text)
        result['arxiv_ids'] = self._extract_arxiv_ids_comprehensive(text)
        result['urls'] = self._extract_urls_comprehensive(text)
        result['emails'] = self._extract_emails_comprehensive(text)
        result['isbns'] = self._extract_isbns(text)
        result['orcids'] = self._extract_orcids(text)
        
        # Count unique identifiers
        unique_dois = self._unique_identifiers(result['dois'])
        unique_arxiv = self._unique_identifiers(result['arxiv_ids'])
        
        result['metadata']['total_identifiers'] = (
            len(result['dois']) + len(result['arxiv_ids']) + 
            len(result['urls']) + len(result['emails']) +
            len(result['isbns']) + len(result['orcids'])
        )
        result['metadata']['unique_dois'] = len(unique_dois)
        result['metadata']['unique_arxiv_ids'] = len(unique_arxiv)
        
        return result
    
    def _extract_dois_comprehensive(self, text: str) -> List[Dict]:
        """
        Extract DOIs from text with full context and validation.
        
        DOI formats supported:
        - https://doi.org/10.xxxx/xxxx
        - http://dx.doi.org/10.xxxx/xxxx
        - doi:10.xxxx/xxxx
        - DOI:10.xxxx/xxxx
        - 10.xxxx/xxxx (standalone)
        """
        import re
        
        dois = []
        
        # Comprehensive DOI patterns
        doi_patterns = [
            # URL-based DOIs
            r'https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s]+)',
            r'https?://(?:dx\.)?doi\.org/doi:(10\.\d{4,}/[^\s]+)',
            
            # Text-based DOIs
            r'(?:doi[:\s]*|DOI[:\s]*|DOI\s+)(10\.\d{4,}/[^\s,]+)',
            r'\(doi[:\s]*?(10\.\d{4,}/[^\s,]+)\)',
            r'\[doi[:\s]?(10\.\d{4,}/[^\s,\]]+)',
            
            # Standalone DOIs (most flexible, placed last)
            r'\b(10\.\d{4,}/[^\s,;)\]]+)\b',
        ]
        
        seen_dois = set()
        
        for pattern in doi_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                doi = match.group(1)
                
                # Clean DOI
                doi = self._clean_doi(doi)
                
                if doi and doi not in seen_dois:
                    seen_dois.add(doi)
                    
                    # Get context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Validate DOI
                    is_valid = self._validate_doi(doi)
                    
                    dois.append({
                        'doi': doi,
                        'context': context.strip(),
                        'position': match.start(),
                        'is_valid': is_valid,
                        'format': self._detect_doi_format(doi)
                    })
        
        return dois
    
    def _clean_doi(self, doi: str) -> str:
        """
        Clean and normalize DOI string.
        """
        import re
        
        # Remove trailing punctuation
        doi = re.sub(r'[.,;)\]]+$', '', doi)
        
        # Remove HTML tags
        doi = re.sub(r'<[^>]+>', '', doi)
        
        # Remove trailing path indicators
        doi = doi.split('?')[0]
        
        # Ensure proper DOI prefix
        if not doi.startswith('10.'):
            # Try to find DOI in the string
            match = re.search(r'(10\.\d{4,}/.+)', doi)
            if match:
                doi = match.group(1)
        
        return doi.strip()
    
    def _validate_doi(self, doi: str) -> bool:
        """
        Validate DOI format and checksum.
        """
        import re
        
        # Basic format validation
        pattern = r'^10\.\d{4,}/[^\s]+$'
        if not re.match(pattern, doi):
            return False
        
        # DOI prefix should be 10.xxxx where xxxx is 4+ digits
        parts = doi.split('/')
        if len(parts) < 2:
            return False
        
        prefix = parts[0]
        suffix = '/'.join(parts[1:])
        
        # Check prefix format
        if not re.match(r'^10\.\d{4,}$', prefix):
            return False
        
        # Suffix should not be empty
        if not suffix or len(suffix) < 1:
            return False
        
        return True
    
    def _detect_doi_format(self, doi: str) -> str:
        """
        Detect the format/variant of a DOI.
        """
        if 'dx.doi.org' in doi:
            return 'dx_doi_org'
        elif 'doi.org' in doi:
            return 'doi_org'
        elif doi.startswith('10.'):
            return 'bare'
        else:
            return 'unknown'
    
    def _extract_arxiv_ids_comprehensive(self, text: str) -> List[Dict]:
        """
        Extract arXiv IDs from text with full context and validation.
        
        arXiv ID formats:
        - arXiv:2101.12345
        - arXiv:2101.12345v1
        - arXiv:2101.12345v2
        - 2101.12345 (bare)
        - 2101.12345v1 (with version)
        - Old style: hep-th/1234567
        """
        import re
        
        arxiv_ids = []
        
        seen_ids = set()
        
        # Comprehensive arXiv patterns
        arxiv_patterns = [
            # URL-based arXiv IDs
            r'https?://arxiv\.org/(?:abs|pdf|ps)/(10\.\d{4,})',
            r'https?://arxiv\.org/abs/(arXiv:\d{4}\.\d+)',
            
            # Text-based with arXiv: prefix
            r'(?:arXiv|ar[xX]iv)[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'(?:arXiv|ar[xX]iv)\s*[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',
            
            # Bare arXiv IDs (more careful to avoid false positives)
            r'(?:^|\s)(\d{4}\.\d{4,5}(?:v\d+)?)(?:\s|[.,;):]|$)',
            
            # Old style arXiv IDs
            r'(?:arXiv|ar[xX]iv)[:\s]*(hep-[a-z]+/\d+)',
            r'(?:^|\s)(hep-[a-z]+/\d{7})(?:\s|[.,;):]|$)',
            
            # arXiv identifier with square brackets
            r'\[ar[xX]iv:(\d{4}\.\d{4,5}(?:v\d+)?)\]',
        ]
        
        for pattern in arxiv_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw_id = match.group(1)
                
                # Normalize to standard format
                arxiv_id = self._normalize_arxiv_id(raw_id)
                
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    
                    # Get context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Determine ID category
                    id_type = self._classify_arxiv_id(arxiv_id)
                    
                    # Validate
                    is_valid = self._validate_arxiv_id(arxiv_id)
                    
                    arxiv_ids.append({
                        'id': arxiv_id,
                        'full_id': f"arXiv:{arxiv_id}",
                        'context': context.strip(),
                        'position': match.start(),
                        'is_valid': is_valid,
                        'type': id_type,
                        'version': self._extract_arxiv_version(arxiv_id),
                        'year': self._extract_arxiv_year(arxiv_id)
                    })
        
        return arxiv_ids
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize arXiv ID to standard format.
        """
        import re
        
        # Remove 'arXiv:' or 'arXiv:' prefix if present
        arxiv_id = re.sub(r'^[aA][rR][xX]iv[:\s]*', '', arxiv_id)
        arxiv_id = re.sub(r'^[aA][rR][xX][iI][vV]:', '', arxiv_id)
        
        # Clean version suffix
        arxiv_id = arxiv_id.strip()
        
        # Validate basic format
        if re.match(r'^\d{4}\.\d{4,5}(?:v\d+)?$', arxiv_id):
            return arxiv_id
        
        if re.match(r'^hep-[a-z]+/\d+$', arxiv_id, re.IGNORECASE):
            return arxiv_id
        
        # Try to extract valid ID
        match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', arxiv_id)
        if match:
            return match.group(1)
        
        match = re.search(r'(hep-[a-z]+/\d+)', arxiv_id, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return ''
    
    def _classify_arxiv_id(self, arxiv_id: str) -> str:
        """
        Classify arXiv ID by format/type.
        """
        import re
        
        # New style (YYMM.NNNNN)
        if re.match(r'^\d{4}\.\d{4,5}$', arxiv_id):
            return 'new_style'
        
        # With version
        if re.match(r'^\d{4}\.\d{4,5}v\d+$', arxiv_id):
            return 'new_style_with_version'
        
        # Old style
        if re.match(r'^[a-z]+-\w+/\d+$', arxiv_id, re.IGNORECASE):
            return 'old_style'
        
        return 'unknown'
    
    def _validate_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate arXiv ID format.
        """
        import re
        
        # New style validation
        if re.match(r'^\d{4}\.\d{4,5}(?:v\d+)?$', arxiv_id):
            return True
        
        # Old style validation
        if re.match(r'^[a-z]+-\w+/\d+$', arxiv_id, re.IGNORECASE):
            return True
        
        return False
    
    def _extract_arxiv_version(self, arxiv_id: str) -> Optional[int]:
        """
        Extract version number from arXiv ID.
        """
        import re
        
        match = re.search(r'v(\d+)$', arxiv_id)
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_arxiv_year(self, arxiv_id: str) -> Optional[int]:
        """
        Extract year from arXiv ID (first 4 digits).
        """
        import re
        
        match = re.match(r'^(\d{4})', arxiv_id)
        if match:
            year = int(match.group(1))
            if 1990 <= year <= 2030:  # Reasonable year range
                return year
        
        return None
    
    def _extract_urls_comprehensive(self, text: str) -> List[Dict]:
        """
        Extract URLs from text with context and classification.
        """
        import re
        
        urls = []
        seen_urls = set()
        
        # URL patterns
        url_patterns = [
            # Standard URLs
            r'https?://[^\s<>""\'()\[\]]+',
            r'ftp://[^\s<>""\'()\[\]]+',
            r'www\.[^\s<>""\'()\[\]]+',
        ]
        
        url_types = {
            'arxiv': r'arxiv\.org',
            'doi': r'doi\.org|dx\.doi\.org',
            'github': r'github\.com',
            'pdf': r'\.pdf$',
            'journal': r'journals?\.|proceedings|ieeexplore|springer|nature|science',
        }
        
        for pattern in url_patterns:
            for match in re.finditer(pattern, text):
                url = match.group(0)
                
                # Clean URL
                url = url.rstrip('.,;')
                
                if url not in seen_urls:
                    seen_urls.add(url)
                    
                    # Classify URL
                    url_type = 'general'
                    for type_name, type_pattern in url_types.items():
                        if re.search(type_pattern, url, re.IGNORECASE):
                            url_type = type_name
                            break
                    
                    # Get context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    urls.append({
                        'url': url,
                        'type': url_type,
                        'context': context.strip(),
                        'position': match.start(),
                        'length': len(url)
                    })
        
        return urls
    
    def _extract_emails_comprehensive(self, text: str) -> List[Dict]:
        """
        Extract email addresses with context.
        """
        import re
        
        emails = []
        seen_emails = set()
        
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        for match in re.finditer(email_pattern, text):
            email = match.group(0)
            
            if email not in seen_emails:
                seen_emails.add(email)
                
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                
                # Validate email format
                is_valid = bool(re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email))
                
                emails.append({
                    'email': email,
                    'context': context.strip(),
                    'position': match.start(),
                    'is_valid': is_valid
                })
        
        return emails
    
    def _extract_isbns(self, text: str) -> List[Dict]:
        """
        Extract ISBN numbers from text.
        """
        import re
        
        isbns = []
        seen = set()
        
        # ISBN-10 and ISBN-13 patterns
        isbn_patterns = [
            r'ISBN[-: ]*(\d{10})',
            r'ISBN[-: ]*(\d{13})',
            r'(\d{3}-\d-\d{4}-\d{4}-\d)',  # ISBN-13 with dashes
            r'(\d{3}-\d-\d{4}-\d{3}-\d)',  # ISBN-10 with dashes
        ]
        
        for pattern in isbn_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                isbn = match.group(1)
                
                if isbn not in seen:
                    seen.add(isbn)
                    
                    isbns.append({
                        'isbn': isbn,
                        'type': 'ISBN-13' if len(isbn) == 13 else 'ISBN-10',
                        'position': match.start()
                    })
        
        return isbns
    
    def _extract_orcids(self, text: str) -> List[Dict]:
        """
        Extract ORCID IDs from text.
        """
        import re
        
        orcids = []
        seen = set()
        
        # ORCID pattern (16-digit format with dashes)
        orcid_pattern = r'\b(\d{4}-\d{4}-\d{4}-\d{4})\b'
        
        for match in re.finditer(orcid_pattern, text):
            orcid = match.group(1)
            
            if orcid not in seen:
                seen.add(orcid)
                
                # Validate checksum
                is_valid = self._validate_orcid(orcid)
                
                orcids.append({
                    'orcid': orcid,
                    'url': f"https://orcid.org/{orcid}",
                    'position': match.start(),
                    'is_valid': is_valid
                })
        
        return orcids
    
    def _validate_orcid(self, orcid: str) -> bool:
        """
        Validate ORCID checksum.
        """
        import re
        
        if not re.match(r'^\d{4}-\d{4}-\d{4}-\d{4}$', orcid):
            return False
        
        # Remove dashes
        orcid_digits = orcid.replace('-', '')
        
        # Last digit is checksum
        check_digit = int(orcid_digits[-1])
        research_digit = int(orcid_digits[-1])
        
        # Calculate weighted sum
        total = 0
        for i, digit in enumerate(orcid_digits[:-1]):
            digit = int(digit)
            if (i + 1) % 2 == 1:
                total += digit * 2
            else:
                total += digit
        
        # Calculate check digit
        remainder = total % 11
        calculated = (11 - remainder) % 11
        
        return calculated == research_digit
    
    def _unique_identifiers(self, identifiers: List[Dict]) -> set:
        """
        Extract unique identifiers from list.
        """
        unique = set()
        for ident in identifiers:
            if 'doi' in ident:
                unique.add(ident['doi'])
            elif 'id' in ident:
                unique.add(ident['id'])
            elif 'email' in ident:
                unique.add(ident['email'])
        return unique
    
    def _format_identifiers_markdown(self, identifiers: Dict) -> str:
        """
        Format extracted identifiers as Markdown.
        """
        if not identifiers or identifiers['metadata']['total_identifiers'] == 0:
            return ''
        
        md_lines = ['\n## Extracted Identifiers\n']
        
        # DOIs
        if identifiers.get('dois'):
            md_lines.append('### DOIs\n')
            for doi in identifiers['dois']:
                status = '✓' if doi['is_valid'] else '✗'
                md_lines.append(f"- {status} [{doi['doi']}](https://doi.org/{doi['doi']})")
                md_lines.append(f"  - Format: {doi['format']}")
            md_lines.append('')
        
        # arXiv IDs
        if identifiers.get('arxiv_ids'):
            md_lines.append('### arXiv IDs\n')
            for arxiv in identifiers['arxiv_ids']:
                status = '✓' if arxiv['is_valid'] else '✗'
                version = arxiv.get('version')
                version_str = f"v{version}" if version else ''
                md_lines.append(f"- {status} [{arxiv['full_id']}](https://arxiv.org/abs/{arxiv['id']}) {version_str}")
                md_lines.append(f"  - Type: {arxiv['type']}")
                if arxiv.get('year'):
                    md_lines.append(f"  - Year: {arxiv['year']}")
            md_lines.append('')
        
        # URLs
        if identifiers.get('urls'):
            url_counts = {}
            for url in identifiers['urls']:
                url_type = url.get('type', 'general')
                url_counts[url_type] = url_counts.get(url_type, 0) + 1
            
            md_lines.append('### URLs\n')
            for url_type, count in sorted(url_counts.items()):
                md_lines.append(f"- {url_type}: {count}")
            md_lines.append('')
        
        # Summary
        md_lines.append('### Summary\n')
        md_lines.append(f"- Total identifiers: {identifiers['metadata']['total_identifiers']}")
        md_lines.append(f"- Unique DOIs: {identifiers['metadata']['unique_dois']}")
        md_lines.append(f"- Unique arXiv IDs: {identifiers['metadata']['unique_arxiv_ids']}")
        
        return '\n'.join(md_lines)
    
    def _detect_table_blocks(self, blocks: List) -> List[List[List[str]]]:
        """Detect table-like structures from text blocks."""
        # This is a simplified table detection - groups blocks that might form tables
        # For better results, use pdfplumber which has dedicated table extraction
        tables = []
        # Group blocks by similar Y coordinates (potential table rows)
        rows = {}
        for block in blocks:
            if len(block) >= 5:
                y_pos = int(block[1] / 10)  # Group by 10-pixel bands
                if y_pos not in rows:
                    rows[y_pos] = []
                rows[y_pos].append(block)
        
        # If we have multiple rows with multiple blocks, might be a table
        if len(rows) >= 2:
            sorted_rows = sorted(rows.items())
            # Check if rows have similar number of blocks (table structure)
            block_counts = [len(blocks) for _, blocks in sorted_rows]
            if len(set(block_counts)) <= 2 and max(block_counts) >= 2:  # Consistent structure
                # Extract as table
                table = []
                for _, row_blocks in sorted_rows:
                    row = []
                    for block in sorted(row_blocks, key=lambda b: b[0]):  # Sort by X
                        if len(block) >= 5:
                            row.append(block[4].strip())  # block[4] is text
                    if row:
                        table.append(row)
                if len(table) >= 2:  # At least 2 rows
                    tables.append(table)
        
        return tables
    
    def _format_tables_markdown(self, tables: List[List[List[str]]]) -> str:
        """Format extracted tables as Markdown for better readability."""
        if not tables:
            return ''
        
        formatted_tables = []
        for table_idx, table in enumerate(tables, 1):
            if not table or len(table) == 0:
                continue
            
            # Format as Markdown table
            table_lines = [f'\n[Table {table_idx}]\n']
            
            # Determine number of columns
            max_cols = max(len(row) for row in table if row) if table else 0
            if max_cols == 0:
                continue
            
            # Format header row (first row)
            if len(table) > 0:
                header = table[0]
                # Pad header to max_cols
                header = header + [''] * (max_cols - len(header))
                # Clean and format header cells
                header_cells = [str(cell).strip().replace('|', '\\|') if cell else '' for cell in header]
                table_lines.append('| ' + ' | '.join(header_cells) + ' |')
                table_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            
            # Format data rows
            for row in table[1:]:
                # Pad row to max_cols
                row = row + [''] * (max_cols - len(row))
                # Clean and format cells
                row_cells = [str(cell).strip().replace('|', '\\|') if cell else '' for cell in row]
                table_lines.append('| ' + ' | '.join(row_cells) + ' |')
            
            formatted_tables.append('\n'.join(table_lines))
        
        return '\n\n'.join(formatted_tables)
    
    def _reconstruct_text_from_dict(self, text_dict: Dict) -> str:
        """Reconstruct text from PyMuPDF dict format with proper ordering."""
        if not text_dict or 'blocks' not in text_dict:
            return ''
        
        lines = []
        for block in text_dict['blocks']:
            if 'lines' not in block:
                continue
            
            block_lines = []
            for line in block['lines']:
                if 'spans' not in line:
                    continue
                
                line_text = []
                for span in line['spans']:
                    if 'text' in span:
                        text = span['text'].strip()
                        if text:
                            line_text.append(text)
                
                if line_text:
                    # Join spans in line with appropriate spacing
                    line_str = ' '.join(line_text)
                    block_lines.append(line_str)
            
            if block_lines:
                lines.extend(block_lines)
        
        return '\n'.join(lines)
    
    def _reconstruct_text_from_blocks(self, blocks: List) -> str:
        """
        Reconstruct text from PyMuPDF blocks with sophisticated multi-column handling.
        
        This method:
        1. Detects number of columns using clustering analysis
        2. Handles mixed single/multi-column layouts
        3. Preserves correct reading order across columns
        4. Processes columns in proper left-to-right, top-to-bottom order
        """
        if not blocks:
            return ''
        
        # Filter and prepare blocks
        valid_blocks = self._prepare_blocks(blocks)
        if not valid_blocks:
            return ''
        
        # Detect column structure
        column_info = self._detect_column_structure(valid_blocks)
        
        # Handle different layout types
        if column_info['is_multi_column']:
            # Multi-column layout - process columns separately then merge
            text = self._process_multi_column_layout(valid_blocks, column_info)
        else:
            # Single column or mixed layout
            text = self._process_single_column_layout(valid_blocks)
        
        # Post-process to fix any remaining line break issues
        text = self._fix_line_breaks(text)
        
        return text
    
    def _prepare_blocks(self, blocks: List) -> List[Dict]:
        """
        Prepare and validate blocks for processing.
        """
        valid_blocks = []
        for block in blocks:
            if len(block) < 5:
                continue
            
            text = block[4].strip() if len(block) > 4 and block[4] else ''
            if not text:
                continue
            
            # Block structure from PyMuPDF: [x0, y0, x1, y1, "text", ...]
            block_info = {
                'x0': block[0],
                'y0': block[1],
                'x1': block[2],
                'y1': block[3],
                'text': text,
                'width': block[2] - block[0],
                'height': block[3] - block[1],
                'area': (block[2] - block[0]) * (block[3] - block[1]),
            }
            
            # Calculate center point
            block_info['center_x'] = (block[0] + block[2]) / 2
            block_info['center_y'] = (block[1] + block[3]) / 2
            
            valid_blocks.append(block_info)
        
        return valid_blocks
    
    def _detect_column_structure(self, blocks: List[Dict]) -> Dict:
        """
        Detect the column structure of the document using clustering analysis.
        
        Returns:
            Dict with:
            - is_multi_column: bool indicating multi-column layout
            - column_count: estimated number of columns
            - column_boundaries: list of (start_x, end_x) for each column
            - gaps: estimated horizontal gaps between columns
        """
        if not blocks:
            return {'is_multi_column': False, 'column_count': 1, 'column_boundaries': [], 'gaps': []}
        
        # Calculate document bounds
        min_x = min(b['x0'] for b in blocks)
        max_x = max(b['x1'] for b in blocks)
        min_y = min(b['y0'] for b in blocks)
        max_y = max(b['y1'] for b in blocks)
        
        doc_width = max_x - min_x
        doc_height = max_y - min_y
        
        # Sample blocks to analyze horizontal distribution
        # Take blocks from different vertical positions
        sample_blocks = self._sample_blocks_vertically(blocks, num_samples=20)
        
        # Analyze horizontal positions
        x_positions = [b['center_x'] for b in sample_blocks]
        x_positions.sort()
        
        # Detect gaps in horizontal distribution (indicates column boundaries)
        gaps = self._detect_horizontal_gaps(x_positions, doc_width)
        
        # Determine column count
        column_count = len(gaps) + 1
        
        # Heuristic: consider it multi-column if:
        # 1. More than one column detected
        # 2. Document is wide enough relative to height
        # 3. Gap size is significant
        
        is_multi_column = False
        aspect_ratio = doc_width / max(doc_height, 1)
        
        if column_count >= 2:
            # Check if columns are well-separated
            avg_gap = sum(g['size'] for g in gaps) / len(gaps) if gaps else 0
            min_gap_threshold = doc_width * 0.05  # 5% of width
            
            if avg_gap > min_gap_threshold or column_count >= 2:
                is_multi_column = True
        
        # If document is in portrait orientation, likely single column
        if aspect_ratio < 0.8 and column_count == 2:
            # Might be single column with margin, not multi-column
            is_multi_column = False
        
        # Calculate column boundaries
        column_boundaries = self._calculate_column_boundaries(blocks, column_count, gaps)
        
        return {
            'is_multi_column': is_multi_column,
            'column_count': column_count,
            'column_boundaries': column_boundaries,
            'gaps': gaps,
            'doc_width': doc_width,
            'doc_height': doc_height,
            'min_x': min_x,
            'max_x': max_x,
        }
    
    def _sample_blocks_vertically(self, blocks: List[Dict], num_samples: int = 20) -> List[Dict]:
        """
        Sample blocks evenly across vertical positions.
        """
        if len(blocks) <= num_samples:
            return blocks
        
        # Sort by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: b['y0'])
        
        # Sample evenly
        step = len(sorted_blocks) / num_samples
        sampled = []
        for i in range(num_samples):
            idx = int(i * step)
            if idx < len(sorted_blocks):
                sampled.append(sorted_blocks[idx])
        
        return sampled
    
    def _detect_horizontal_gaps(self, x_positions: List[float], doc_width: float) -> List[Dict]:
        """
        Detect gaps in horizontal block distribution.
        Returns list of gaps with position and size.
        """
        if len(x_positions) < 4:
            return []
        
        gaps = []
        
        # Calculate gaps between consecutive x positions
        for i in range(len(x_positions) - 1):
            gap_size = x_positions[i + 1] - x_positions[i]
            gap_center = (x_positions[i] + x_positions[i + 1]) / 2
            
            gaps.append({
                'start': x_positions[i],
                'end': x_positions[i + 1],
                'center': gap_center,
                'size': gap_size,
            })
        
        # Find significant gaps (larger than average)
        if not gaps:
            return []
        
        avg_gap = sum(g['size'] for g in gaps) / len(gaps)
        significant_gaps = [g for g in gaps if g['size'] > avg_gap * 2]
        
        # Also check for very large gaps (clear column separators)
        threshold = doc_width * 0.1  # 10% of document width
        very_large_gaps = [g for g in gaps if g['size'] > threshold]
        
        # Combine significant and very large gaps
        final_gaps = list(set(g['center'] for g in significant_gaps + very_large_gaps))
        final_gaps.sort()
        
        # Convert to gap dicts
        result = []
        for center in final_gaps:
            # Find the gap that contains this center
            for g in gaps:
                if g['center'] == center:
                    result.append(g)
                    break
        
        return result
    
    def _calculate_column_boundaries(self, blocks: List[Dict], column_count: int, 
                                     gaps: List[Dict]) -> List[Tuple[float, float]]:
        """
        Calculate the x-boundaries for each column.
        """
        if column_count == 1:
            min_x = min(b['x0'] for b in blocks)
            max_x = max(b['x1'] for b in blocks)
            return [(min_x, max_x)]
        
        # Use gap positions to determine boundaries
        gap_centers = [g['center'] for g in gaps]
        gap_centers.sort()
        
        boundaries = []
        
        if column_count == 2:
            # Two columns: left of gap and right of gap
            if gap_centers:
                gap_x = gap_centers[0]
                left_max = min(b['x1'] for b in blocks if b['center_x'] < gap_x)
                right_min = max(b['x0'] for b in blocks if b['center_x'] > gap_x)
                boundaries.append((0, left_max))
                boundaries.append((right_min, max(b['x1'] for b in blocks)))
            else:
                # Split document in half
                doc_min = min(b['x0'] for b in blocks)
                doc_max = max(b['x1'] for b in blocks)
                mid = (doc_min + doc_max) / 2
                boundaries.append((doc_min, mid))
                boundaries.append((mid, doc_max))
        
        else:
            # More than 2 columns - distribute evenly
            doc_min = min(b['x0'] for b in blocks)
            doc_max = max(b['x1'] for b in blocks)
            col_width = (doc_max - doc_min) / column_count
            for i in range(column_count):
                start = doc_min + i * col_width
                end = doc_min + (i + 1) * col_width
                boundaries.append((start, end))
        
        return boundaries
    
    def _process_multi_column_layout(self, blocks: List[Dict], column_info: Dict) -> str:
        """
        Process multi-column layout by extracting each column in proper order.
        
        Reading order for multi-column:
        1. Left column, top to bottom
        2. Right column, top to bottom
        (Not: left column, then right column page by page)
        """
        boundaries = column_info['column_boundaries']
        if not boundaries:
            return self._process_single_column_layout(blocks)
        
        # Assign each block to a column
        column_blocks = [[] for _ in range(len(boundaries))]
        
        for block in blocks:
            col_idx = self._get_column_index(block, boundaries)
            if col_idx >= 0:
                column_blocks[col_idx].append(block)
        
        # Process each column's blocks in proper order
        column_texts = []
        for col_idx, col_blocks in enumerate(column_blocks):
            if col_blocks:
                # Sort blocks within column top to bottom
                sorted_blocks = sorted(col_blocks, key=lambda b: b['y0'])
                column_text = self._blocks_to_text(sorted_blocks)
                column_texts.append(column_text)
        
        # Merge columns with proper spacing
        # For multi-column papers, columns should be processed left-to-right
        merged_text = '\n\n'.join(column_texts)
        
        return merged_text
    
    def _get_column_index(self, block: Dict, boundaries: List[Tuple[float, float]]) -> int:
        """
        Determine which column a block belongs to.
        """
        center_x = block['center_x']
        
        for idx, (start, end) in enumerate(boundaries):
            if start <= center_x < end:
                return idx
        
        # Block doesn't fit neatly - assign to closest column
        if center_x < boundaries[0][0]:
            return 0
        elif center_x >= boundaries[-1][1]:
            return len(boundaries) - 1
        else:
            # Find closest boundary
            min_dist = float('inf')
            closest_idx = 0
            for idx, (start, end) in enumerate(boundaries):
                col_center = (start + end) / 2
                dist = abs(center_x - col_center)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            return closest_idx
    
    def _process_single_column_layout(self, blocks: List[Dict]) -> str:
        """
        Process single-column or simple layout.
        """
        # Sort blocks top to bottom, left to right
        sorted_blocks = sorted(blocks, key=lambda b: (b['y0'], b['x0']))
        
        return self._blocks_to_text(sorted_blocks)
    
    def _blocks_to_text(self, blocks: List[Dict]) -> str:
        """
        Convert sorted blocks to text with proper spacing.
        """
        if not blocks:
            return ''
        
        text_parts = []
        prev_block = None
        
        for block in blocks:
            block_text = block['text']
            if not block_text:
                continue
            
            # Add spacing based on block positions
            if prev_block:
                x_gap = block['x0'] - prev_block['x1']
                y_gap = block['y0'] - prev_block['y1']
                
                # Determine spacing
                if y_gap > 30:  # Significant vertical gap - new paragraph
                    text_parts.append('\n\n')
                elif y_gap > 10:  # Moderate vertical gap - new line
                    text_parts.append('\n')
                elif x_gap > 10:  # Horizontal gap within line
                    text_parts.append(' ')
            
            text_parts.append(block_text)
            prev_block = block
        
        return ''.join(text_parts)
    
    def _detect_mixed_layout(self, blocks: List[Dict]) -> Dict:
        """
        Detect if document has mixed layout (some pages single, some multi-column).
        """
        if not blocks:
            return {'has_mixed_layout': False}
        
        # Analyze blocks in different vertical regions
        regions = self._divide_into_regions(blocks, num_regions=5)
        
        column_counts = []
        for region_blocks in regions.values():
            col_info = self._detect_column_structure(region_blocks)
            column_counts.append(col_info['column_count'])
        
        # Check if column count varies
        unique_counts = set(column_counts)
        
        if len(unique_counts) > 1:
            return {
                'has_mixed_layout': True,
                'region_column_counts': dict(zip(regions.keys(), column_counts)),
                'dominant_count': max(set(column_counts), key=column_counts.count),
            }
        
        return {'has_mixed_layout': False}
    
    def _divide_into_regions(self, blocks: List[Dict], num_regions: int = 5) -> Dict[str, List[Dict]]:
        """
        Divide blocks into horizontal regions for analysis.
        """
        if not blocks:
            return {}
        
        min_y = min(b['y0'] for b in blocks)
        max_y = max(b['y1'] for b in blocks)
        height = max_y - min_y
        
        regions = {}
        region_height = height / num_regions
        
        for i in range(num_regions):
            region_start = min_y + i * region_height
            region_end = region_start + region_height
            regions[f'region_{i}'] = [b for b in blocks if region_start <= b['y0'] < region_end]
        
        return regions
    
    def _handle_mixed_layout(self, blocks: List[Dict]) -> str:
        """
        Handle documents with mixed column layouts.
        """
        # Process each region separately with its detected column count
        regions = self._divide_into_regions(blocks, num_regions=10)
        
        region_texts = []
        prev_region_end_y = 0
        
        for region_name in sorted(regions.keys()):
            region_blocks = regions[region_name]
            if not region_blocks:
                continue
            
            # Detect column structure for this region
            col_info = self._detect_column_structure(region_blocks)
            
            # Process region
            if col_info['is_multi_column']:
                region_text = self._process_multi_column_layout(region_blocks, col_info)
            else:
                region_text = self._process_single_column_layout(region_blocks)
            
            region_texts.append(region_text)
        
        # Merge regions with proper spacing
        return '\n\n'.join(region_texts)
    
    def _fix_line_breaks(self, text: str) -> str:
        """
        Fix common line break issues in reconstructed text.
        
        This method handles:
        1. Broken words across lines (with hyphens and without)
        2. Single characters on lines
        3. Improper paragraph breaks
        4. Multi-column artifacts
        """
        if not text:
            return text
        
        # Fix hyphenated words broken across lines
        # Pattern: word- followed by newline and more word characters
        text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
        
        # Fix non-hyphenated word breaks across lines
        # Only join if it looks like a broken word (lowercase followed by lowercase)
        text = re.sub(r'([a-z])\s*\n\s*([a-z][a-z]+)', r'\1\2', text)
        
        # Fix single character lines that are likely artifacts
        # But preserve intentional single characters (initials, math symbols, etc.)
        lines = text.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            # Skip empty lines for now (handle later)
            if not line:
                fixed_lines.append('')
                continue
            
            # Check if line is a single character or very short
            if len(line) == 1:
                # Check context: is it surrounded by text on both sides?
                prev_line = fixed_lines[-1] if fixed_lines else ''
                # We'll check next line later, for now preserve single chars
                fixed_lines.append(line)
            elif len(line) <= 3 and i > 0 and i < len(lines) - 1:
                # Very short line - check if it should join with previous or next
                prev_content = fixed_lines[-1].strip() if fixed_lines else ''
                next_content = lines[i + 1].strip() if i + 1 < len(lines) else ''
                
                # If previous line ends with lowercase and next line starts with lowercase,
                # this is likely a broken word
                if prev_content and next_content:
                    if prev_content[-1].islower() and next_content[0].islower():
                        # Join with previous line
                        fixed_lines[-1] = prev_content + line
                        continue
                
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Rejoin lines
        text = '\n'.join(fixed_lines)
        
        # Fix multiple consecutive line breaks (normalize to max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix improper paragraph breaks
        # If a line ends with a sentence-ending punctuation followed by a short line,
        # and the next line starts with uppercase, it's likely a new paragraph
        lines = text.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                fixed_lines.append('')
                continue
            
            # Check if this line should be joined with previous
            if i > 0:
                prev_line = fixed_lines[-1].strip()
                if prev_line:
                    # If previous line ends with sentence-ending punctuation
                    # and this line starts with uppercase (likely new sentence, not new para)
                    if prev_line[-1] in '.!?' and line and line[0].isupper():
                        # This is a continuation, not a new paragraph
                        fixed_lines[-1] = prev_line + ' ' + line
                        continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _extract_and_format_tables_pdfplumber(self, tables: List, page) -> str:
        """Extract and format tables from pdfplumber with enhanced formatting."""
        if not tables:
            return ''
        
        formatted_tables = []
        
        for table_idx, table in enumerate(tables, 1):
            if not table:
                continue
            
            # Try to get table bounding box for caption detection
            # Note: Disabled find_tables to avoid stderr noise
            # Table bounding box detection disabled to prevent find_tables exceptions
            table_bbox = None
            
            # Extract table caption (text above table)
            caption = None
            if table_bbox:
                try:
                    # Get text above table (within 50 pixels)
                    words_above = page.extract_words()
                    for word in words_above:
                        if (word['top'] < table_bbox[1] and 
                            word['top'] > table_bbox[1] - 50 and
                            abs(word['left'] - table_bbox[0]) < 100):
                            # Potential caption
                            if caption is None:
                                caption = word['text']
                            else:
                                caption += ' ' + word['text']
                except:
                    pass
            
            # Format table as Markdown
            table_lines = []
            if caption:
                table_lines.append(f'\n[Table {table_idx}: {caption}]\n')
            else:
                table_lines.append(f'\n[Table {table_idx}]\n')
            
            # Determine number of columns
            max_cols = max(len(row) for row in table if row) if table else 0
            if max_cols == 0:
                continue
            
            # Format header row (first row, if it looks like a header)
            has_header = False
            if len(table) > 0:
                first_row = table[0]
                # Check if first row looks like a header (short text, all caps, etc.)
                if first_row:
                    first_row_text = ' '.join(str(cell) for cell in first_row if cell).strip()
                    # Heuristic: if first row is short and has few words, likely header
                    if len(first_row_text) < 100 and len(first_row_text.split()) < 10:
                        has_header = True
                        header = first_row + [''] * (max_cols - len(first_row))
                        header_cells = [str(cell).strip().replace('|', '\\|') if cell else '' for cell in header]
                        table_lines.append('| ' + ' | '.join(header_cells) + ' |')
                        table_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            
            # Format data rows
            start_idx = 1 if has_header else 0
            for row in table[start_idx:]:
                if not row:
                    continue
                # Pad row to max_cols
                row = row + [''] * (max_cols - len(row))
                # Clean and format cells
                row_cells = [str(cell).strip().replace('|', '\\|').replace('\n', ' ') if cell else '' for cell in row]
                table_lines.append('| ' + ' | '.join(row_cells) + ' |')
            
            formatted_tables.append('\n'.join(table_lines))
        
        return '\n\n'.join(formatted_tables)

    # ============================================================
    # Character Encoding and Text Normalization
    # ============================================================

    def _fix_encoding_issues(self, text: str, preserve_math: bool = True) -> str:
        """
        Fix common encoding issues in extracted text.

        Handles:
        - Unicode replacement characters
        - Ligature expansion (ﬁ → fi, etc.)
        - Special quotation marks and dashes
        - Mathematical symbols
        - Accented characters ( normalization)
        - Control characters
        - Common encoding artifacts

        Args:
            text: Text with potential encoding issues
            preserve_math: Whether to preserve mathematical symbols

        Returns:
            Cleaned text with encoding issues fixed
        """
        import unicodedata

        if not text:
            return text

        # Step 1: Remove control characters (except common whitespace)
        # Keep: tab (9), newline (10), carriage return (13), space (32)
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\t\n\r')

        # Step 2: Replace Unicode replacement character with empty string
        text = text.replace('\ufffd', '')
        text = text.replace('\x00', '')  # Also remove null bytes

        # Step 3: Normalize Unicode using NFKC (compatibility decomposition)
        text = unicodedata.normalize('NFKC', text)

        # Step 4: Expand ligatures (keep some for special cases)
        ligature_map = {
            'ﬁ': 'fi',   # Latin small ligature fi
            'ﬂ': 'fl',   # Latin small ligature fl
            'ﬀ': 'ff',   # Latin small ligature ff
            'ﬃ': 'ffi',  # Latin small ligature ffi
            'ﬄ': 'ffl',  # Latin small ligature ffl
            'ﬅ': 'st',   # Latin small letter long s
            'ﬆ': 'st',   # Latin small ligature st
            'ᵢ': 'i',    # Subscript i
            'ⱼ': 'j',    # Subscript j
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)

        # Step 5: Normalize smart quotes and dashes
        quote_map = {
            # Double quotes
            '"': '"',   # Left double quotation mark
            '"': '"',   # Right double quotation mark
            '"': '"',   # Double high-reversed quotation mark
            '"': '"',   # Double low-reversed quotation mark
            # Single quotes
            ''': "'",   # Left single quotation mark
            ''': "'",   # Right single quotation mark
            ''': "'",   # Single high-reversed quotation mark
            ''': "'",   # Single low-reversed quotation mark
            # Apostrophe variants
            ''': "'",   # Modifier letter prime
            ''': "'",   # Modifier letter double prime
            # Dashes and hyphens
            '\u2010': '-',  # Hyphen
            '\u2011': '-',  # Non-breaking hyphen
            '\u2012': '-',  # Figure dash
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2015': '-',  # Horizontal bar
            '\u2212': '-',  # Minus sign
            # Ellipsis
            '…': '...',  # Horizontal ellipsis
            '‥': '..',   # Two dot leader
        }
        for smart, normal in quote_map.items():
            text = text.replace(smart, normal)

        # Step 6: Handle mathematical symbols (keep if preserve_math)
        if not preserve_math:
            math_map = {
                '×': 'x',      # Multiplication sign
                '÷': '/',      # Division sign
                '±': '+/-',    # Plus-minus
                '∓': '-/+',    # Minus-plus
                '≤': '<=',     # Less than or equal
                '≥': '>=',     # Greater than or equal
                '≠': '!=',     # Not equal
                '≈': '~',      # Approximately equal
                '≡': '==',     # Identical
                '∞': 'infinity',  # Infinity
                '√': 'sqrt',   # Square root
                '∑': 'sum',    # Summation
                '∏': 'prod',   # Product
                '∂': 'd',      # Partial derivative
                '∇': 'grad',   # Gradient
                '∆': 'delta',  # Delta
                '→': '->',     # Arrow right
                '←': '<-',     # Arrow left
                '↔': '<->',    # Arrow both
                '⇒': '=>',     # Implication
                '⇔': '<=>',    # Equivalence
            }
            for symbol, replacement in math_map.items():
                text = text.replace(symbol, replacement)

        # Step 7: Handle common encoding artifacts
        # Zero-width characters
        text = text.replace('\u200b', '')  # Zero width space
        text = text.replace('\u200c', '')  # Zero width non-joiner
        text = text.replace('\u200d', '')  # Zero width joiner
        text = text.replace('\u200e', '')  # Left-to-right mark
        text = text.replace('\u200f', '')  # Right-to-left mark

        # Step 8: Handle subscript/superscript digits (convert to normal)
        subscript_map = {
            '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
            '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
        }
        superscript_map = {
            '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
            '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
            '\u2078': '8', '\u2079': '9',
        }
        for sub, normal in subscript_map.items():
            text = text.replace(sub, normal)
        for sup, normal in superscript_map.items():
            text = text.replace(sup, normal)

        # Step 9: Fix common OCR errors
        ocr_fixes = {
            '|l': 'I',   # Vertical bar l should be I
            'I0': 'IO',  # I0 might be IO
            '0O': 'OO',  # 0O might be OO
            'rn': 'm',   # rn might be m
        }
        for error, correction in ocr_fixes.items():
            text = text.replace(error, correction)

        # Step 10: Clean up multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def _handle_rtl_text(self, text: str, mark_rtl: bool = True) -> str:
        """
        Handle right-to-left text (Arabic, Hebrew, etc.).

        This method:
        1. Detects RTL script sections
        2. Properly orients text for reading
        3. Adds directional markers if requested

        Args:
            text: Text that may contain RTL sections
            mark_rtl: Whether to wrap RTL sections with RLE...PDF markers

        Returns:
            Text with RTL sections properly handled
        """
        import unicodedata

        if not text:
            return text

        # RTL script categories
        rtl_scripts = {
            'Hebrew': '\u0590-\u05FF',
            'Arabic': '\u0600-\u06FF',
            'Syriac': '\u0700-\u074F',
            'Thaana': '\u0780-\u07BF',
            'Nko': '\u07C0-\u07FF',
            'Samaritan': '\u0800-\u083F',
            'Mandaic': '\u0840-\u085F',
            'Arabic Supplement': '\u0750-\u077F',
        }

        # Check if text contains RTL characters
        has_rtl = False
        rtl_chars = []
        for script_name, char_range in rtl_scripts.items():
            for char in text:
                if '\u0590' <= char <= '\u05FF' or \
                   '\u0600' <= char <= '\u06FF' or \
                   '\u0700' <= char <= '\u074F':
                    has_rtl = True
                    rtl_chars.append(char)
                    break
            if has_rtl:
                break

        if not has_rtl:
            return text

        # For academic papers, RTL text (like Arabic abstracts) should often
        # be marked but not reversed, as PDFs usually contain pre-oriented text

        if mark_rtl:
            # Wrap detected RTL sections with directional markers
            # RLE = Right-to-Left Embedding
            # PDF = Pop Directional Formatting

            # Find RTL paragraphs/sections and wrap them
            lines = text.split('\n')
            processed_lines = []

            for line in lines:
                # Check if line contains RTL characters
                line_has_rtl = False
                for char in line:
                    if '\u0590' <= char <= '\u05FF' or \
                       '\u0600' <= char <= '\u06FF' or \
                       '\u0700' <= char <= '\u074F':
                        line_has_rtl = True
                        break

                if line_has_rtl:
                    # Wrap RTL line with directional markers
                    processed_lines.append(f'\u202A{line}\u202C')
                else:
                    processed_lines.append(line)

            return '\n'.join(processed_lines)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters in text.

        Handles:
        - Multiple spaces → single space
        - Tabs → spaces
        - Non-breaking spaces → regular spaces
        - Various space variants
        """
        if not text:
            return text

        # Replace various space characters with regular space
        space_chars = [
            '\t',      # Tab
            '\u00A0',  # Non-breaking space
            '\u2000',  # En quad
            '\u2001',  # Em quad
            '\u2002',  # En space
            '\u2003',  # Em space
            '\u2004',  # Three-per-em space
            '\u2005',  # Four-per-em space
            '\u2006',  # Six-per-em space
            '\u2007',  # Figure space
            '\u2008',  # Punctuation space
            '\u2009',  # Thin space
            '\u200A',  # Hair space
            '\u202F',  # Narrow no-break space
            '\u205F',  # Medium mathematical space
        ]

        for char in space_chars:
            text = text.replace(char, ' ')

        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)

        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _fix_special_characters(self, text: str) -> str:
        """
        Fix special characters commonly corrupted in PDF extraction.

        Handles:
        - Accented characters
        - Currency symbols
        - Commercial symbols
        - Fraction characters
        - Roman numerals
        - Misc symbols
        """
        if not text:
            return text

        # Fraction mapping
        fraction_map = {
            '½': '1/2',
            '⅓': '1/3',
            '⅔': '2/3',
            '¼': '1/4',
            '¾': '3/4',
            '⅕': '1/5',
            '⅖': '2/5',
            '⅗': '3/5',
            '⅘': '4/5',
            '⅙': '1/6',
            '⅚': '5/6',
            '⅐': '1/7',
            '⅛': '1/8',
            '⅜': '3/8',
            '⅝': '5/8',
            '⅞': '7/8',
            '⅑': '1/9',
            '⅒': '1/10',
        }

        # Currency symbols
        currency_map = {
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY/CNY',
            '₹': 'INR',
            '₩': 'KRW',
            '₽': 'RUB',
            '¢': 'cent',
            '₣': 'FRF',
            '₤': 'GBP',
            '₦': 'NGN',
            '₧': 'ESP',
            '₨': 'INR/PKR',
            '₪': 'ILS',
            '₫': 'VND',
            '₭': 'LAK',
            '₮': 'MNT',
            '₯': 'GRD',
            '₰': 'PF',
            '₱': 'PHP',
            '₲': 'PYG',
            '₳': 'ARA',
            '₴': 'UAH',
            '₵': 'GHS',
            '₸': 'KZT',
            '₺': 'TRY',
            '₼': 'AZN',
            '₾': 'GEL',
        }

        # Roman numerals
        roman_map = {
            'Ⅰ': 'I',
            'Ⅱ': 'II',
            'Ⅲ': 'III',
            'Ⅳ': 'IV',
            'Ⅴ': 'V',
            'Ⅵ': 'VI',
            'Ⅶ': 'VII',
            'Ⅷ': 'VIII',
            'Ⅸ': 'IX',
            'Ⅹ': 'X',
            'Ⅺ': 'XI',
            'Ⅻ': 'XII',
            'Ⅼ': 'L',
            'Ⅽ': 'C',
            'Ⅾ': 'D',
            'Ⅿ': 'M',
        }

        # Arrow symbols
        arrow_map = {
            '←': '<-',
            '→': '->',
            '↑': '^',
            '↓': 'v',
            '↔': '<->',
            '⇐': '<=',
            '⇒': '=>',
            '⇔': '<=>',
        }

        # Apply mappings
        for old, new in fraction_map.items():
            text = text.replace(old, new)
        for old, new in currency_map.items():
            text = text.replace(old, new)
        for old, new in roman_map.items():
            text = text.replace(old, new)
        for old, new in arrow_map.items():
            text = text.replace(old, new)

        return text

    def _extract_figures_with_ocr(self, pdf_path: str) -> List[Dict]:
        """
        Extract figures and generate descriptions using OCR.

        Returns:
            List of figure dictionaries with images and descriptions
        """
        if not self.enable_ocr:
            logger.debug("OCR disabled, skipping figure extraction")
            return []

        figures = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"Could not open PDF for figure extraction: {e}")
            return []

        for page_num, page in enumerate(doc):
            # Detect image regions
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]

                # Get image coordinates
                img_rect = self._find_image_bounding_box(page, xref)

                if img_rect is None:
                    continue

                # Extract image
                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue
                    image_bytes = base_image["image"]
                except Exception as e:
                    logger.debug(f"Could not extract image {img_index}: {e}")
                    continue

                # Save temporary for OCR
                try:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                        tmp.write(image_bytes)
                        tmp.flush()

                        # Analyze image content
                        analysis = self._analyze_figure_content(tmp_path)

                        # Generate description based on figure type
                        if analysis['is_chart']:
                            description = self._describe_chart_with_vision(tmp_path)
                        else:
                            description = self._extract_text_from_image(tmp_path)

                        figures.append({
                            'page': page_num + 1,
                            'image_index': img_index,
                            'bbox': [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1],
                            'type': analysis['type'],
                            'description': description,
                            'has_text': analysis['has_text'],
                            'confidence': analysis['confidence']
                        })

                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

                except Exception as e:
                    logger.debug(f"Error processing image {img_index}: {e}")
                    continue

        doc.close()
        return figures

    def _find_image_bounding_box(self, page, xref: int) -> Optional[fitz.Rect]:
        """Find the bounding box for an image on a page."""
        try:
            # Get image list with bounding boxes
            image_list = page.get_images(full=True)

            for img in image_list:
                if img[0] == xref:
                    # Try to get the bounding box
                    # PyMuPDF doesn't directly provide bbox for images,
                    # so we use a heuristic based on image properties
                    base_image = page.parent.extract_image(xref)
                    if base_image:
                        # Calculate approximate bounding box
                        # This is a best-effort estimate
                        width = base_image.get('width', 0)
                        height = base_image.get('height', 0)

                        if width > 0 and height > 0:
                            # Get page dimensions
                            page_rect = page.rect
                            # Estimate image position (usually centered or inline)
                            # Return a reasonable default rectangle
                            return fitz.Rect(0, 0, min(width, page_rect.width), min(height, page_rect.height))

            return None
        except Exception as e:
            logger.debug(f"Error finding image bounding box: {e}")
            return None

    def _analyze_figure_content(self, image_path: str) -> Dict:
        """Analyze figure type using image processing."""
        import cv2
        import numpy as np
        import tempfile

        analysis = {
            'is_chart': False,
            'type': 'unknown',
            'has_text': False,
            'confidence': 0.0
        }

        try:
            img = cv2.imread(image_path)
            if img is None:
                return analysis

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Check image dimensions
            if gray.shape[0] < 50 or gray.shape[1] < 50:
                # Image too small, likely not a significant figure
                return analysis

            # Detect various figure types using multiple heuristics
            # 1. Check for lines (charts, graphs)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

            if lines is not None and len(lines) > 3:
                analysis['is_chart'] = True
                analysis['type'] = 'line_chart'
                analysis['confidence'] = min(0.9, 0.5 + (len(lines) / 100))

            # 2. Check for bars (bar charts)
            if not analysis['is_chart']:
                # Threshold for bar detection
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
                horizontal_lines = cv2.countNonZero(horizontal_lines)

                if horizontal_lines > 100:
                    analysis['is_chart'] = True
                    analysis['type'] = 'bar_chart'
                    analysis['confidence'] = 0.7

            # 3. Check for scatter plots (many small points)
            if not analysis['is_chart']:
                # Detect circles/blobs
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                           minDist=20, param1=50, param2=15,
                                           minRadius=2, maxRadius=20)
                if circles is not None and len(circles[0]) > 10:
                    analysis['is_chart'] = True
                    analysis['type'] = 'scatter_plot'
                    analysis['confidence'] = 0.6

            # 4. Check for pie chart-like circular structures
            if not analysis['is_chart']:
                max_radius = min(100, min(gray.shape[:2]) // 2)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                           minDist=50, param1=50, param2=30,
                                           minRadius=30, maxRadius=max_radius)
                if circles is not None:
                    analysis['is_chart'] = True
                    analysis['type'] = 'pie_chart'
                    analysis['confidence'] = 0.6

            # Check for text regions
            text_regions = self._detect_text_regions(gray)
            analysis['has_text'] = len(text_regions) > 0

            # If no specific chart type detected but has many edges, might be a diagram
            if not analysis['is_chart']:
                edge_density = cv2.countNonZero(edges) / (gray.shape[0] * gray.shape[1])
                if edge_density > 0.1:
                    analysis['is_chart'] = True
                    analysis['type'] = 'diagram'
                    analysis['confidence'] = 0.5

        except Exception as e:
            logger.debug(f"Error analyzing figure content: {e}")

        return analysis

    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect text regions in an image using contour analysis."""
        import cv2

        regions = []

        try:
            # Apply morphological operations to enhance text regions
            # Increase kernel sizes for better text detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            dilated = cv2.dilate(gray_image, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out small contours (likely noise)
                if w > 50 and h > 15:
                    aspect_ratio = w / h if h > 0 else 0
                    # Text regions typically have wider aspect ratios
                    if 1 < aspect_ratio < 20:
                        regions.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'aspect_ratio': aspect_ratio
                        })

        except Exception as e:
            logger.debug(f"Error detecting text regions: {e}")

        return regions

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        if not self.enable_ocr:
            return "[Image with text - OCR disabled]"

        try:
            text = pytesseract.image_to_string(image_path, lang=self.ocr_language)
            if text.strip():
                return f"[Image text]: {text.strip()}"
            else:
                return "[Image - no readable text detected]"
        except Exception as e:
            logger.debug(f"OCR text extraction failed: {e}")
            return "[Image - OCR extraction failed]"

    def _describe_chart_with_vision(self, image_path: str) -> str:
        """Generate a description of a chart/diagram using OCR analysis."""
        if not self.enable_ocr:
            return "[Chart/Diagram - OCR disabled for description]"

        try:
            # Extract structured information from the chart
            # 1. Get text elements
            data = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT)

            # 2. Analyze text elements
            texts = []
            for i, txt in enumerate(data['text']):
                if int(data['conf'][i]) > 30 and txt.strip():
                    texts.append(txt.strip())

            # 3. Detect axis labels if present
            axis_texts = []
            for text in texts:
                # Common axis label patterns
                if any(pattern in text.lower() for pattern in ['fig', 'figure', 'x:', 'y:', 'axis', 'label']):
                    axis_texts.append(text)

            # 4. Generate description
            if axis_texts:
                description = f"[Chart/Diagram with labels: {', '.join(axis_texts[:5])}]"
            elif texts:
                # Use detected text as description
                description = f"[Chart/Diagram containing: {'; '.join(texts[:3])}]"
            else:
                description = "[Chart/Diagram - visual content detected]"

            return description

        except Exception as e:
            logger.debug(f"Chart description failed: {e}")
            return "[Chart/Diagram - description extraction failed]"

    def _extract_pdfplumber(self, pdf_path: str) -> Optional[Dict]:
        """Extract using pdfplumber (better for tables and complex layouts)."""
        with pdfplumber.open(pdf_path) as pdf:
            text_pages = []
            full_text = []
            
            metadata = {
                'title': pdf.metadata.get('Title', '') if pdf.metadata else '',
                'author': pdf.metadata.get('Author', '') if pdf.metadata else '',
                'subject': pdf.metadata.get('Subject', '') if pdf.metadata else '',
                'creator': pdf.metadata.get('Creator', '') if pdf.metadata else '',
                'producer': pdf.metadata.get('Producer', '') if pdf.metadata else '',
                'page_count': len(pdf.pages)
            }
            
            # Collect all tables from all pages
            all_tables = []
            
            for page_num, page in enumerate(pdf.pages):
                page_text = ''
                
                # Strategy 1: Try layout mode first (better for multi-column)
                # Maximum quality: always try layout mode
                try:
                    page_text_layout = page.extract_text(layout=True) or ''
                    if page_text_layout:
                        page_text = page_text_layout
                except Exception as e:
                    logger.debug(f"pdfplumber layout extraction failed: {e}")
                
                # Strategy 2: Standard extraction as comparison
                # Maximum quality: always try standard extraction
                try:
                    page_text_standard = page.extract_text() or ''
                    if page_text_standard and (not page_text or len(page_text_standard) > len(page_text) * 1.02):
                        page_text = page_text_standard
                except Exception as e:
                    logger.debug(f"pdfplumber standard extraction failed: {e}")
                
                # Strategy 3: Word-level extraction with improved ordering (for complex layouts)
                # Maximum quality: always try word-level extraction
                try:
                    words = page.extract_words(
                        x_tolerance=3,  # Horizontal tolerance for word grouping
                        y_tolerance=3,  # Vertical tolerance for word grouping
                        keep_blank_chars=False,
                        use_text_flow=True  # Use text flow for better ordering
                    )
                    if words:
                        # Improved sorting: group by lines first, then sort within lines
                        words_by_line = {}
                        for word in words:
                            # Group words by vertical position (same line)
                            y_pos = int(word.get('top', 0) / 5)  # 5-pixel tolerance
                            if y_pos not in words_by_line:
                                words_by_line[y_pos] = []
                            words_by_line[y_pos].append(word)
                        
                        # Sort lines and words within lines
                        sorted_words = []
                        for y_pos in sorted(words_by_line.keys()):
                            line_words = sorted(words_by_line[y_pos], key=lambda w: w.get('left', 0))
                            sorted_words.extend(line_words)
                        
                        # Reconstruct text with proper spacing
                        page_text_words = self._reconstruct_text_from_words(sorted_words)
                        if page_text_words and len(page_text_words) > len(page_text) * 0.9:
                            page_text = page_text_words
                except Exception as e:
                    logger.debug(f"pdfplumber word extraction failed: {e}")
                
                # Strategy 4: Extract tables with enhanced formatting
                try:
                    tables = page.extract_tables()
                    if tables:
                        # Collect tables for return value
                        all_tables.extend(tables)
                        # Enhanced table extraction with better formatting
                        table_text = self._extract_and_format_tables_pdfplumber(tables, page)
                        if table_text:
                            if page_text:
                                page_text += '\n\n' + table_text
                            else:
                                page_text = table_text
                except Exception as e:
                    logger.debug(f"pdfplumber table extraction failed: {e}")
                
                text_pages.append({
                    'page': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
                full_text.append(page_text)
            
            # Post-process the full text for better quality
            full_text_combined = '\n\n'.join(full_text)
            full_text_combined = self._post_process_extracted_text(full_text_combined)
            
        return {
            'text': full_text_combined,
            'metadata': metadata,
            'pages': text_pages,
            'method_used': 'pdfplumber',
            # Include extracted tables (list of 2-D arrays) if any were found
            'tables': all_tables,
        }
    
    def _reconstruct_text_from_words(self, words: List[Dict]) -> str:
        """Reconstruct text from word list with proper spacing."""
        if not words:
            return ''
        
        text_parts = []
        prev_word = None
        
        for word in words:
            word_text = word.get('text', '').strip()
            if not word_text:
                continue
            
            if prev_word:
                # Calculate spacing based on positions
                x_gap = word.get('left', 0) - (prev_word.get('left', 0) + prev_word.get('width', 0))
                y_gap = abs(word.get('top', 0) - prev_word.get('top', 0))
                
                # Add newline if significant vertical movement
                if y_gap > prev_word.get('height', 10) * 0.5:
                    text_parts.append('\n')
                # Add space if horizontal gap suggests word boundary
                elif x_gap > 2:  # More than 2 pixels gap
                    text_parts.append(' ')
            
            text_parts.append(word_text)
            prev_word = word
        
        return ''.join(text_parts)
    
    def _extract_pypdf(self, pdf_path: str) -> Optional[Dict]:
        """Extract using pypdf (lightweight fallback) with improved extraction."""
        reader = PdfReader(pdf_path)
        text_pages = []
        full_text = []
        
        metadata = {}
        if reader.metadata:
            metadata = {
                'title': reader.metadata.get('/Title', ''),
                'author': reader.metadata.get('/Author', ''),
                'subject': reader.metadata.get('/Subject', ''),
                'creator': reader.metadata.get('/Creator', ''),
                'producer': reader.metadata.get('/Producer', ''),
            }
        metadata['page_count'] = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            # Try standard extraction first
            try:
                page_text = page.extract_text() or ''
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                page_text = ''
            
            # If extraction is poor, try multiple strategies
            if len(page_text) < 1000:  # Try alternatives if extraction seems incomplete
                # Strategy 1: Try with layout mode
                if hasattr(page, 'extract_text'):
                    try:
                        page_text_alt = page.extract_text(extraction_mode="layout") or ''
                        if len(page_text_alt) > len(page_text) * 1.1:
                            page_text = page_text_alt
                    except Exception as e:
                        logger.debug(f"Layout mode failed for page {page_num + 1}: {e}")
                        pass
                
                # Strategy 2: Try extracting text with different parameters
                if len(page_text) < 500:
                    try:
                        # Try extracting with different approach (if available in pypdf version)
                        # Some versions support different extraction modes
                        if hasattr(page, 'extract_text'):
                            # Try without any mode specification
                            try:
                                page_text_alt2 = str(page.extract_text()) or ''
                                if len(page_text_alt2) > len(page_text):
                                    page_text = page_text_alt2
                            except:
                                pass
                    except:
                        pass
            
            # Clean up common pypdf extraction issues
            # Fix broken words (common in pypdf) - but be careful with acronyms
            # Only fix if it looks like a word break (lowercase followed by uppercase in middle of sentence)
            page_text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', page_text)  # Fix word breaks
            # Fix broken numbers (but preserve intentional spaces)
            page_text = re.sub(r'(\d)\s+(\d)', r'\1\2', page_text)  # Fix number breaks
            # Fix broken punctuation
            page_text = re.sub(r'([a-zA-Z])\s+([.,;:!?])', r'\1\2', page_text)  # Fix punctuation spacing
            # Fix broken quotes
            page_text = re.sub(r'([a-zA-Z])\s+(["\'])', r'\1\2', page_text)
            page_text = re.sub(r'(["\'])\s+([a-zA-Z])', r'\1\2', page_text)
            
            text_pages.append({
                'page': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text.append(page_text)
        
        # Post-process the full text for better quality
        full_text_combined = '\n\n'.join(full_text)
        full_text_combined = self._post_process_extracted_text(full_text_combined)
        
        return {
            'text': full_text_combined,
            'metadata': metadata,
            'pages': text_pages,
            'method_used': 'pypdf'
        }
    
    def _extract_ocr(self, pdf_path: str) -> Optional[Dict]:
        """Extract using OCR (for scanned/image-based PDFs) with adaptive PSM modes."""
        try:
            # Maximum quality: Use high DPI for best OCR quality
            images = convert_from_path(pdf_path, dpi=400)  # Increased from 300 for maximum quality
        except Exception as e:
            logger.warning(f"Failed to convert PDF to images: {e}")
            return None
        
        text_pages = []
        full_text = []
        
        for page_num, image in enumerate(images):
            page_text = ''
            
            # Preprocess image for better OCR
            try:
                from PIL import Image, ImageEnhance, ImageFilter
                
                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')
                
                # Maximum quality image enhancement
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.8)  # Increased contrast for better OCR
                
                # Sharpen image
                image = image.filter(ImageFilter.SHARPEN)
                
                # Additional enhancement for maximum quality
                enhancer_brightness = ImageEnhance.Brightness(image)
                image = enhancer_brightness.enhance(1.1)  # Slight brightness boost
            except Exception as e:
                logger.debug(f"Image preprocessing failed for page {page_num + 1}: {e}")
            
            # Maximum quality: Try all PSM modes and use the best result
            psm_modes = [
                ('6', 'Uniform block of text'),  # Default for most academic papers
                ('3', 'Fully automatic page segmentation'),  # Good for complex layouts
                ('4', 'Single column of text'),  # Good for single column
                ('11', 'Sparse text'),  # Good for sparse text
                ('1', 'Automatic page segmentation with OSD'),  # Additional mode for quality
                ('7', 'Single text line'),  # For line-by-line extraction
            ]
            
            best_text = ''
            best_length = 0
            
            for psm_mode, description in psm_modes:
                try:
                    # Maximum quality OCR config with additional parameters
                    whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}\'"-+=*/<>%&|\\@#$^_~`'
                    # Add quality-enhancing OCR parameters
                    config_str = f'--psm {psm_mode} -c tessedit_char_whitelist={whitelist} --oem 3 -c preserve_interword_spaces=1'
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang=self.ocr_language,
                        config=config_str
                    )
                    
                    # Post-process OCR text
                    ocr_text = self._post_process_ocr_text(ocr_text)
                    
                    # Use the mode that extracts the most text (with quality check)
                    if len(ocr_text) > best_length:
                        # Basic quality check: should have reasonable word count
                        words = ocr_text.split()
                        if len(words) > 10:  # At least 10 words
                            best_text = ocr_text
                            best_length = len(ocr_text)
                except Exception as e:
                    logger.debug(f"OCR PSM {psm_mode} failed for page {page_num + 1}: {e}")
                    continue
            
            # If no good result, try default mode
            if not best_text:
                try:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=self.ocr_language,
                        config='--psm 6'
                    )
                    page_text = self._post_process_ocr_text(page_text)
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    page_text = ''
            else:
                page_text = best_text
            
            text_pages.append({
                'page': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text.append(page_text)
        
        # Post-process the full text
        full_text_combined = '\n\n'.join(full_text)
        full_text_combined = self._post_process_extracted_text(full_text_combined)
        
        return {
            'text': full_text_combined,
            'metadata': {'page_count': len(images), 'ocr_used': True},
            'pages': text_pages,
            'method_used': 'ocr'
        }
    
    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text to fix common OCR errors."""
        if not text:
            return text
        
        # Fix spacing issues (OCR often has inconsistent spacing)
        # But preserve intentional line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        
        # Fix broken line breaks (OCR sometimes breaks words incorrectly)
        text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)  # Lowercase-lowercase across lines
        
        # Fix common OCR character confusions (conservative approach)
        # Fix "rn" -> "m" when it's clearly wrong (between lowercase letters)
        text = re.sub(r'([a-z])rn([a-z])', r'\1m\2', text)
        # Fix "vv" -> "w" when it's clearly wrong
        text = re.sub(r'([a-z])vv([a-z])', r'\1w\2', text)
        # Fix "ii" -> "n" when it's clearly wrong (less common, be careful)
        text = re.sub(r'([a-z])ii([a-z])', r'\1n\2', text)
        
        # Fix common OCR spacing issues around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after punctuation
        
        # Fix broken numbers (OCR sometimes breaks numbers)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Remove spaces in numbers
        
        # Fix broken URLs and emails
        text = re.sub(r'([a-zA-Z0-9])\s+([@.])\s+([a-zA-Z0-9])', r'\1\2\3', text)
        
        # Normalize quotes (OCR sometimes uses wrong quote characters)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Fix broken hyphens (OCR sometimes breaks hyphenated words)
        text = re.sub(r'([a-zA-Z])\s*-\s*\n\s*([a-zA-Z])', r'\1-\2', text)
        
        return text
    
    def _detect_pdf_type(self, pdf_path: str) -> str:
        """
        Detect PDF type: 'text-based', 'scanned', or 'mixed'.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDF type string
        """
        if not PYMUPDF_AVAILABLE:
            return 'unknown'
        
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                doc.close()
                return 'unknown'
            
            # Sample first few pages to determine type
            sample_pages = min(3, len(doc))
            text_chars = 0
            total_chars = 0
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                # Try to extract text
                page_text = page.get_text()
                text_chars += len(page_text)
                
                # Count total characters (including images)
                # If page has images but little text, likely scanned
                images = page.get_images()
                if images:
                    total_chars += 1000  # Estimate for image-based pages
            
            doc.close()
            
            # If we got substantial text, it's text-based
            if text_chars > 500 * sample_pages:
                return 'text-based'
            # If we got very little text but have images, likely scanned
            elif text_chars < 100 * sample_pages and total_chars > 0:
                return 'scanned'
            # Otherwise, mixed or unknown
            else:
                return 'mixed'
        except Exception as e:
            logger.debug(f"PDF type detection failed: {e}")
            return 'unknown'
    
    # ============================================================
    # Abstract Detection and Structured Extraction
    # ============================================================
    
    def _detect_abstract(self, text: str) -> Dict:
        """
        Detect and extract the abstract from text with validation.
        
        Returns:
            Dict with:
            - has_abstract: bool indicating if abstract was found
            - abstract: the extracted abstract text
            - position: character position where abstract starts
            - confidence: confidence score (0-1)
            - validation: dict with validation results
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'validation': {
                'has_minimum_length': False,
                'has_sentence_structure': False,
                'has_common_terms': False,
                'proper_position': False,
            },
            'extraction_method': None
        }
        
        if not text:
            return result
        
        # Try multiple detection strategies
        strategies = [
            self._detect_abstract_strategy_1,  # Standard "Abstract" header
            self._detect_abstract_strategy_2,  # "1. Abstract" numbered
            self._detect_abstract_strategy_3,  # "Abstract:" with colon
            self._detect_abstract_strategy_4,  # Summary keyword
            self._detect_abstract_strategy_5,  # First page content analysis
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for strategy in strategies:
            strategy_result = strategy(text)
            if strategy_result['confidence'] > best_confidence:
                best_confidence = strategy_result['confidence']
                best_result = strategy_result
        
        if best_result:
            # Validate the extracted abstract
            validation = self._validate_abstract(best_result['abstract'])
            best_result['validation'] = validation
            
            # Calculate final confidence with validation
            validation_score = sum(validation.values()) / len(validation) if validation else 0
            best_result['confidence'] = best_confidence * 0.7 + validation_score * 0.3
            
            # Only accept if validation passes
            if validation_score >= 0.5:
                return best_result
        
        return result
    
    def _detect_abstract_strategy_1(self, text: str) -> Dict:
        """
        Strategy 1: Standard "Abstract" header.
        Pattern: "Abstract" at start of line, followed by text until next section.
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'extraction_method': 'strategy_1'
        }
        
        # Patterns for abstract header
        abstract_patterns = [
            r'^\s*abstract\s*$',
            r'^\s*Abstract\s*$',
            r'^\s*ABSTRACT\s*$',
            r'^Abstract\s*[:\n]',
            r'^\s*abstract\s*[:\n]',
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in abstract_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Found abstract header
                    abstract_lines = []
                    
                    # Collect lines until next section or significant break
                    for j in range(i + 1, min(i + 50, len(lines))):
                        next_line = lines[j].strip()
                        
                        # Check for next section header
                        section_patterns = [
                            r'^\s*\d+\.\s*(?:introduction|background|related\s*work|method|methodology|approach|results?|discussion|conclusion)',
                            r'^\s*(?:introduction|background|related\s*work|methodology?)\s*[:\n]',
                            r'^\s*I\.?\s+',
                            r'^\s*1\s+',
                        ]
                        
                        if any(re.match(p, next_line, re.IGNORECASE) for p in section_patterns):
                            break
                        
                        # Skip empty lines at start
                        if not abstract_lines and not next_line:
                            continue
                        
                        abstract_lines.append(lines[j])
                    
                    if abstract_lines:
                        abstract_text = '\n'.join(abstract_lines).strip()
                        
                        # Check minimum length
                        if len(abstract_text) > 50:
                            result['has_abstract'] = True
                            result['abstract'] = abstract_text
                            result['position'] = text.find(lines[i])
                            result['confidence'] = 0.9  # High confidence for clear header
                    
                    return result
        
        return result
    
    def _detect_abstract_strategy_2(self, text: str) -> Dict:
        """
        Strategy 2: Numbered abstract section.
        Pattern: "1. Abstract" or "I. Abstract" followed by text.
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'extraction_method': 'strategy_2'
        }
        
        # Numbered abstract patterns
        numbered_patterns = [
            r'^\s*1\.?\s*abstract\s*$',
            r'^\s*I\.?\s*abstract\s*$',
            r'^\s*Abstract\s*1\.?\s*$',
            r'^\s*1\s+abstract\s*$',
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in numbered_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    abstract_lines = []
                    
                    # Collect lines until next section
                    for j in range(i + 1, min(i + 50, len(lines))):
                        next_line = lines[j].strip()
                        
                        # Check for next section
                        if re.match(r'^\s*\d+\.', next_line) and j > i + 1:
                            break
                        
                        if not abstract_lines and not next_line:
                            continue
                        
                        abstract_lines.append(lines[j])
                    
                    if abstract_lines:
                        abstract_text = '\n'.join(abstract_lines).strip()
                        
                        if len(abstract_text) > 50:
                            result['has_abstract'] = True
                            result['abstract'] = abstract_text
                            result['position'] = text.find(lines[i])
                            result['confidence'] = 0.85
                    
                    return result
        
        return result
    
    def _detect_abstract_strategy_3(self, text: str) -> Dict:
        """
        Strategy 3: "Abstract:" with colon delimiter.
        Pattern: "Abstract: ... text ..."
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'extraction_method': 'strategy_3'
        }
        
        # Abstract: pattern
        colon_pattern = r'^\s*[Aa]bstract\s*:\s*(.+)$'
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(colon_pattern, line)
            if match:
                abstract_text = match.group(1).strip()
                
                # Add continuation lines
                for j in range(i + 1, min(i + 30, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Stop at next section or double newline
                    if not next_line:
                        continue
                    
                    if re.match(r'^\s*\d+\.', next_line) or next_line.lower() in ['introduction', 'keywords']:
                        break
                    
                    abstract_text += ' ' + next_line
                
                if len(abstract_text) > 50:
                    result['has_abstract'] = True
                    result['abstract'] = abstract_text
                    result['position'] = text.find(line)
                    result['confidence'] = 0.8
                
                return result
        
        return result
    
    def _detect_abstract_strategy_4(self, text: str) -> Dict:
        """
        Strategy 4: Summary/Overview keyword detection.
        Pattern: Uses "summary" or "overview" keywords for papers without clear "Abstract" header.
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'extraction_method': 'strategy_4'
        }
        
        summary_patterns = [
            r'^\s*summary\s*$',
            r'^\s*Summary\s*$',
            r'^\s*overview\s*$',
            r'^\s*Overview\s*$',
            r'^\s*executive\s*summary\s*$',
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in summary_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    abstract_lines = []
                    
                    for j in range(i + 1, min(i + 50, len(lines))):
                        next_line = lines[j].strip()
                        
                        if not abstract_lines and not next_line:
                            continue
                        
                        # Check for next section
                        if re.match(r'^\s*\d+\.', next_line):
                            break
                        
                        abstract_lines.append(lines[j])
                    
                    if abstract_lines:
                        abstract_text = '\n'.join(abstract_lines).strip()
                        
                        if len(abstract_text) > 50:
                            result['has_abstract'] = True
                            result['abstract'] = abstract_text
                            result['position'] = text.find(lines[i])
                            result['confidence'] = 0.6  # Lower confidence - not a true abstract
                    
                    return result
        
        return result
    
    def _detect_abstract_strategy_5(self, text: str) -> Dict:
        """
        Strategy 5: First page content analysis.
        Analyzes the first page to find abstract-like content when explicit header is missing.
        """
        import re
        
        result = {
            'has_abstract': False,
            'abstract': '',
            'position': -1,
            'confidence': 0.0,
            'extraction_method': 'strategy_5'
        }
        
        # Get first 3000 characters (typically contains abstract on page 1)
        first_page = text[:3000]
        lines = first_page.split('\n')
        
        # Skip title and author lines (usually at very top)
        skip_lines = 0
        for i, line in enumerate(lines[:10]):
            # Skip title (often longest line at top)
            # Skip author lines (often contain affiliations)
            if i < 3:
                skip_lines = i + 1
                continue
            
            # Look for paragraph-like text that could be abstract
            line = line.strip()
            if len(line) > 100 and len(line) < 1000:
                # Check for abstract-like content
                abstract_indicators = [
                    r'\b(propose|present|introduce|demonstrate|show|describe|discuss)\b',
                    r'\b(paper|study|research|approach|method|technique)\b',
                    r'\b(results|findings|experiments|evaluation)\b',
                ]
                
                indicator_count = sum(1 for p in abstract_indicators if re.search(p, line, re.IGNORECASE))
                
                if indicator_count >= 2:
                    result['has_abstract'] = True
                    result['abstract'] = line
                    result['position'] = text.find(line)
                    result['confidence'] = 0.5  # Low confidence - heuristic only
                    
                    # Try to extend
                    for j in range(skip_lines + 1, min(skip_lines + 10, len(lines))):
                        ext_line = lines[j].strip()
                        if len(ext_line) > 50 and not re.match(r'^\d+\.', ext_line):
                            result['abstract'] += ' ' + ext_line
                        else:
                            break
                    
                    return result
        
        return result
    
    def _validate_abstract(self, abstract: str) -> Dict:
        """
        Validate extracted abstract for quality and completeness.
        
        Returns:
            Dict with validation results for various quality checks
        """
        import re
        
        validation = {
            'has_minimum_length': False,
            'has_sentence_structure': False,
            'has_common_terms': False,
            'proper_position': False,
        }
        
        if not abstract:
            return validation
        
        # Check minimum length (at least 50 characters)
        validation['has_minimum_length'] = len(abstract) > 50
        
        # Check sentence structure (has periods, question marks, or exclamation marks)
        sentences = re.split(r'[.!?]+', abstract)
        sentences = [s.strip() for s in sentences if s.strip()]
        validation['has_sentence_structure'] = len(sentences) >= 1
        
        # Check for common abstract terms
        abstract_terms = [
            'propose', 'present', 'introduce', 'demonstrate', 'show',
            'describe', 'discuss', 'study', 'research', 'approach',
            'method', 'technique', 'result', 'finding', 'experiment',
            'paper', 'work', 'task', 'problem', 'solution'
        ]
        
        abstract_lower = abstract.lower()
        term_count = sum(1 for term in abstract_terms if term in abstract_lower)
        validation['has_common_terms'] = term_count >= 3
        
        # Check position (should be near start of document)
        # This is set by the caller based on document analysis
        validation['proper_position'] = True  # Assume true if we got this far
        
        return validation
    
    def _extract_document_structure(self, text: str) -> Dict:
        """
        Extract the hierarchical document structure.
        
        Returns:
            Dict with:
            - sections: list of detected sections with positions
            - figures: list of figure references
            - tables: list of table references
            - equations: list of equation references
            - abstract: extracted abstract (if found)
            - outline: hierarchical document outline
        """
        import re
        
        structure = {
            'sections': [],
            'figures': [],
            'tables': [],
            'equations': [],
            'algorithms': [],
            'abstract': None,
            'keywords': [],
            'acknowledgments': None,
            'appendices': [],
            'outline': None,  # Hierarchical structure
        }
        
        if not text:
            return structure
        
        lines = text.split('\n')
        
        # Detect sections with hierarchical numbering using comprehensive patterns
        sections = self._detect_sections_comprehensive(lines)
        structure['sections'] = sections
        
        # Build hierarchical outline
        structure['outline'] = self._build_section_outline(sections)
        
        # Detect figure references
        figure_patterns = [
            r'[Ff]ig(?:ure)?\.?\s*(\d+)',
            r'[Ff]igure\s*(\d+)',
        ]
        
        for pattern in figure_patterns:
            for match in re.finditer(pattern, text):
                structure['figures'].append({
                    'number': match.group(1),
                    'position': match.start()
                })
        
        # Detect table references
        table_patterns = [
            r'[Tt]ab(?:le)?\.?\s*(\d+)',
            r'[Tt]able\s*(\d+)',
        ]
        
        for pattern in table_patterns:
            for match in re.finditer(pattern, text):
                structure['tables'].append({
                    'number': match.group(1),
                    'position': match.start()
                })
        
        # Detect equation references
        eq_patterns = [
            r'\((\d+)\)',
            r'[Ee]q(?:uation)?\.?\s*(\d+)',
            r'\((\d+\.\d+)\)',
        ]
        
        for pattern in eq_patterns:
            for match in re.finditer(pattern, text):
                # Check if it looks like an equation reference (not just a number)
                context_before = text[max(0, match.start() - 20):match.start()]
                if any(kw in context_before.lower() for kw in ['eq', 'equation', 'see', 'in']):
                    structure['equations'].append({
                        'number': match.group(1),
                        'position': match.start()
                    })

        # Detect algorithms and pseudocode
        algorithms = self._extract_algorithms(text)
        structure['algorithms'] = algorithms

        # Detect keywords section
        keywords_pattern = r'^\s*keywords?\s*[:\n](.+)$'
        for i, line in enumerate(lines[:50]):  # Keywords usually near abstract
            match = re.match(keywords_pattern, line, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                # Extract individual keywords
                keywords = re.split(r'[;,]', keywords_text)
                structure['keywords'] = [k.strip() for k in keywords if k.strip()]
                break
        
        return structure

    def _extract_algorithms(self, text: str) -> List[Dict]:
        """
        Detect and extract algorithms/pseudocode from document text.

        Identifies:
        - Algorithm environments and numbered algorithms
        - Pseudocode with keywords
        - Algorithm input/output blocks
        - Function and procedure definitions

        Args:
            text: Document text to search

        Returns:
            List of algorithm dictionaries with structure preserved
        """
        import re

        algorithms = []

        if not text:
            return algorithms

        lines = text.split('\n')

        # Pseudocode keyword patterns for detection
        pseudocode_keywords = {
            'control': ['if', 'else', 'elif', 'for', 'while', 'foreach', 'do', 'repeat', 'until', 'switch', 'case', 'break', 'continue'],
            'structure': ['function', 'procedure', 'return', 'begin', 'end', 'then', 'input', 'output', 'data'],
            'operations': ['set', 'get', 'compute', 'calculate', 'initialize', 'update', 'increment', 'decrement', 'sort', 'search'],
            'logic': ['and', 'or', 'not', 'true', 'false', 'null', 'nil', 'undefined']
        }

        all_keywords = set()
        for category in pseudocode_keywords.values():
            all_keywords.update(category)

        # Algorithm header patterns
        algo_header_patterns = [
            # Standard algorithm headers
            r'^\s*(?:Algorithm|ALGORITHM|Algorithm\s+\d+)\s*[:.]?\s*(.+)$',
            r'^\s*(?:Procedure|PROCEDURE|Procedure\s+\d+)\s*[:.]?\s*(.+)$',
            r'^\s*(?:Function|FUNCTION|Function\s+\d+)\s*[:.]?\s*(.+)$',
            r'^\s*(?:Listing|LISTING|Listing\s+\d+)\s*[:.]?\s*(.+)$',
            r'^\s*(?:Pseudocode|PSEUDOCODE)\s*[:.]?\s*(.+)$',
            # Numbered algorithms: "1. Algorithm X" or "Algorithm 1:"
            r'^\s*\d+\.\s*(?:Algorithm|ALGORITHM|Procedure|PROCEDURE|Function|FUNCTION)\s*[:.]?\s*(.+)$',
            r'^\s*(?:Algorithm|ALGORITHM)\s+(\d+)[:.]?\s*(.+)$',
        ]

        # Input/Output block patterns
        io_patterns = {
            'input': r'^\s*(?:Input|INPUT)\s*[:]\s*(.+)$',
            'output': r'^\s*(?:Output|OUTPUT)\s*[:]\s*(.+)$',
            'requires': r'^\s*(?:Requires|REQUIRES)\s*[:]\s*(.+)$',
            'returns': r'^\s*(?:Returns|RETURNS)\s*[:]\s*(.+)$',
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this is an algorithm header
            algo_match = None
            for pattern in algo_header_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    algo_match = {
                        'type': 'algorithm',
                        'start_line': i,
                        'number': '',
                        'title': '',
                        'lines': [],
                        'input': [],
                        'output': [],
                        'keywords': [],
                        'steps': [],
                        'confidence': 0.0
                    }

                    # Determine number and title from match groups
                    if len(groups) == 1:
                        algo_match['title'] = groups[0].strip() if groups[0] else ''
                    elif len(groups) == 2:
                        # Pattern with explicit number
                        algo_match['number'] = groups[0].strip() if groups[0] else ''
                        algo_match['title'] = groups[1].strip() if groups[1] else ''
                    break

            if algo_match:
                # Collect algorithm block
                algo_lines = []
                j = i + 1

                # First pass: collect input/output blocks
                while j < len(lines):
                    next_line = lines[j].strip()
                    next_line_lower = next_line.lower()

                    # Check for input block
                    input_match = re.match(io_patterns['input'], next_line, re.IGNORECASE)
                    if input_match:
                        algo_match['input'].append(input_match.group(1).strip())
                        j += 1
                        continue

                    # Check for output block
                    output_match = re.match(io_patterns['output'], next_line, re.IGNORECASE)
                    if output_match:
                        algo_match['output'].append(output_match.group(1).strip())
                        j += 1
                        continue

                    # Check for end of algorithm
                    end_patterns = [
                        r'^\s*(?:end|END)\s*(?:algorithm|procedure|function)?',
                        r'^\s*(?:done|DONE|return|RETURN)',
                        r'^\s*(?:\d+\.?\s+[A-Z])',  # Next numbered section
                        r'^\s*(?:Figure|Table|Table)\s+\d+',  # Figure/table reference
                    ]
                    if any(re.match(p, next_line, re.IGNORECASE) for p in end_patterns):
                        break

                    # Check if still algorithm content
                    has_keyword = any(kw in next_line_lower for kw in all_keywords)
                    is_numbered_step = re.match(r'^\s*\d+\.?\s+\w+', next_line)
                    is_comment = next_line.startswith('#') or next_line.startswith('//')
                    is_blank = not next_line

                    if has_keyword or is_numbered_step or is_comment or not is_blank:
                        algo_lines.append(lines[j])
                        j += 1
                    else:
                        # Non-algorithm content, might be end
                        if algo_lines and not is_blank:
                            # Check if next few lines are also non-algorithm
                            look_ahead = min(j + 5, len(lines))
                            non_algo_count = 0
                            for k in range(j, look_ahead):
                                check_line = lines[k].strip().lower()
                                if not check_line:
                                    continue
                                if not any(kw in check_line for kw in all_keywords) and \
                                   not re.match(r'^\s*\d+\.?\s+', check_line):
                                    non_algo_count += 1
                                else:
                                    break

                            if non_algo_count >= 3:
                                break

                        if not is_blank:
                            j += 1
                    # If blank line after content, might be end
                    if is_blank and algo_lines:
                        look_ahead = min(j + 3, len(lines))
                        if all(not lines[k].strip() for k in range(j, look_ahead)):
                            break
                    j += 1

                # Process collected lines
                algo_match['lines'] = algo_lines
                algo_match['text'] = '\n'.join(algo_lines)

                # Extract keywords from algorithm content
                all_text = algo_match['text'].lower()
                found_keywords = set()
                for category, keywords in pseudocode_keywords.items():
                    for kw in keywords:
                        if kw in all_text:
                            found_keywords.add(kw)
                algo_match['keywords'] = sorted(list(found_keywords))

                # Parse numbered steps
                step_pattern = r'^\s*(\d+)\.?\s*(.+)$'
                for algo_line in algo_lines:
                    step_match = re.match(step_pattern, algo_line.strip())
                    if step_match:
                        algo_match['steps'].append({
                            'number': int(step_match.group(1)),
                            'text': step_match.group(2).strip()
                        })

                # Calculate confidence based on keywords and structure
                confidence = 0.3  # Base confidence
                if algo_match['title']:
                    confidence += 0.2
                if len(algo_match['keywords']) >= 3:
                    confidence += 0.2
                if len(algo_match['steps']) >= 2:
                    confidence += 0.2
                if algo_match['input'] or algo_match['output']:
                    confidence += 0.1
                algo_match['confidence'] = min(1.0, confidence)

                # Only include if we have meaningful content
                if len(algo_match['lines']) >= 2:
                    algorithms.append(algo_match)

                i = j
                continue

            # Also check for inline algorithms (functions/procedures in text)
            inline_pattern = r'^\s*(?:function|procedure)\s+(\w+)\s*\(([^)]*)\)'
            inline_match = re.match(inline_pattern, line, re.IGNORECASE)
            if inline_match and not algo_match:
                func_name = inline_match.group(1)
                params = inline_match.group(2).strip()

                # Collect function body
                func_lines = [line]
                func_keywords = ['function', 'procedure']
                brace_count = line.count('{') - line.count('}')
                j = i + 1

                while j < len(lines) and (brace_count > 0 or not lines[j].strip()):
                    func_lines.append(lines[j])
                    brace_count += lines[j].count('{') - lines[j].count('}')

                    # Check for pseudocode keywords
                    line_lower = lines[j].strip().lower()
                    for category in pseudocode_keywords.values():
                        for kw in category:
                            if kw in line_lower:
                                func_keywords.append(kw)
                    j += 1

                if len(func_lines) >= 2:
                    algorithms.append({
                        'type': 'inline_function',
                        'start_line': i,
                        'number': '',
                        'title': f"{func_name}({params})",
                        'lines': func_lines,
                        'input': [params] if params else [],
                        'output': [],
                        'keywords': list(set(func_keywords)),
                        'steps': [],
                        'confidence': 0.6
                    })

            i += 1

        # Remove duplicates (same title within close proximity)
        seen_titles = set()
        unique_algorithms = []
        for algo in algorithms:
            title_key = algo['title'].lower() if algo['title'] else str(algo['start_line'])
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_algorithms.append(algo)

        return unique_algorithms

    def _format_algorithms_for_output(self, algorithms: List[Dict]) -> str:
        """
        Format algorithms for inclusion in output documents.

        Args:
            algorithms: List of extracted algorithms

        Returns:
            Formatted algorithm text with proper structure
        """
        if not algorithms:
            return ''

        lines = []
        lines.append('\n## Algorithms\n')

        for algo in algorithms:
            num = algo.get('number', '')
            title = algo.get('title', 'Unnamed Algorithm')
            algo_type = algo.get('type', 'algorithm')

            if num:
                lines.append(f'### Algorithm {num}: {title}\n')
            else:
                lines.append(f'### {title}\n')

            # Add input/output if available
            if algo.get('input'):
                lines.append('**Input:** ' + ', '.join(algo['input']))
            if algo.get('output'):
                lines.append('**Output:** ' + ', '.join(algo['output']))

            # Add algorithm body with line numbers
            lines.append('\n```\n')
            for idx, line in enumerate(algo.get('lines', []), 1):
                lines.append(f'{idx:3d} | {line}')
            lines.append('```\n')

            # Add keywords
            if algo.get('keywords'):
                lines.append(f'*Keywords:* {", ".join(sorted(algo["keywords"]))}\n')

            # Add confidence
            confidence = algo.get('confidence', 0)
            lines.append(f'*Confidence:* {confidence:.2%}\n')

        return '\n'.join(lines)

    def _detect_algorithm_captions(self, text: str) -> List[Dict]:
        """
        Detect algorithm captions in the text.

        Returns:
            List of algorithm captions with their positions
        """
        import re

        captions = []

        caption_patterns = [
            r'(?:[Aa]lgorithm|[Pp]rocedure|[Ll]isting)\s*(\d+)[:.]\s*([^\n]+)',
            r'(?:[Aa]lgorithm|[Pp]rocedure)\s*(\d+(?:\.\d+)*)\s*[:]\s*([^\n]+)',
        ]

        for pattern in caption_patterns:
            for match in re.finditer(pattern, text):
                captions.append({
                    'number': match.group(1),
                    'text': match.group(2).strip() if match.lastindex >= 2 else '',
                    'position': match.start()
                })

        return captions

    def _extract_paper_metadata(self, text: str, metadata: Dict = None) -> Dict:
        """
        Extract paper metadata including title, authors, abstract, and structure.
        
        Args:
            text: Full paper text
            metadata: Existing metadata dict (from PDF extraction)
            
        Returns:
            Enhanced metadata dict with extracted fields
        """
        if metadata is None:
            metadata = {}
        
        # Extract document structure
        structure = self._extract_document_structure(text)
        
        # Add structure to metadata
        metadata['document_structure'] = structure
        
        # Add abstract if found
        if structure.get('abstract'):
            metadata['abstract'] = structure['abstract']['abstract']
            metadata['abstract_confidence'] = structure['abstract']['confidence']
            metadata['abstract_validation'] = structure['abstract']['validation']
        
        # Add keywords if found
        if structure.get('keywords'):
            metadata['keywords'] = structure['keywords']
        
        # Add section count
        metadata['section_count'] = len(structure.get('sections', []))
        
        # Add figure/table counts
        metadata['figure_count'] = len(structure.get('figures', []))
        metadata['table_count'] = len(structure.get('tables', []))
        
        # Detect paper type from structure
        paper_type = self._detect_paper_type(structure)
        metadata['paper_type'] = paper_type
        
        return metadata
    
    def _detect_paper_type(self, structure: Dict) -> str:
        """
        Detect the type of paper based on its structure.
        
        Types:
        - 'research_paper': Standard research paper with intro/methods/results/conclusion
        - 'survey': Survey or review paper
        - 'short_paper': Short paper or poster
        - 'technical_report': Technical report
        - 'unknown': Could not determine
        """
        sections = structure.get('sections', [])
        if not sections:
            return 'unknown'
        
        section_titles = ' '.join(s.get('title', '') for s in sections).lower()
        
        # Check for survey indicators
        survey_indicators = ['survey', 'review', 'taxonomy', 'literature review']
        if any(ind in section_titles for ind in survey_indicators):
            return 'survey'
        
        # Check for research paper indicators
        research_indicators = ['introduction', 'method', 'result', 'conclusion', 'experiment']
        indicator_count = sum(1 for ind in research_indicators if ind in section_titles)
        if indicator_count >= 3:
            return 'research_paper'
        
        # Check for short paper (few sections)
        if len(sections) <= 4:
            return 'short_paper'
        
        # Check for technical report
        if 'technical report' in section_titles or 'technical report' in section_titles:
            return 'technical_report'
        
        return 'research_paper'  # Default assumption
    
    # ============================================================
    # Section Structure Preservation
    # ============================================================
    
    def _detect_sections_comprehensive(self, lines: List[str]) -> List[Dict]:
        """
        Comprehensive section detection with multiple pattern strategies.
        
        Detects:
        - Numbered sections (1, 1.1, 1.1.1, A, B, C)
        - Unnumbered sections (ALL CAPS, bold headers)
        - Special sections (Appendix, References, etc.)
        """
        import re
        
        sections = []
        
        # Comprehensive section patterns
        section_patterns = [
            # Arabic numbered: 1. Introduction
            (r'^\s*(\d+)\.\s+(.+)$', 1, 'arabic'),
            
            # Sub-sections: 1.1 Methods, 1.1.1 Data
            (r'^\s*(\d+(?:\.\d+)+)\.\s+(.+)$', 2, 'arabic_sub'),
            
            # Roman numerals: I. Introduction, II. Related Work
            (r'^\s*(I{1,3}|IV|V|VI{0,3})\.\s+(.+)$', 1, 'roman'),
            
            # Letters: A. Background, B. Methodology
            (r'^\s*([A-Z])\.\s+(.+)$', 1, 'letter'),
            
            # Unnumbered: ALL CAPS headers
            (r'^\s*([A-Z][A-Z0-9\s&\-]{5,})\s*$', 0, 'uppercase'),
            
            # Numbered with parentheses: (1) Introduction
            (r'^\s*\(\s*(\d+)\s*\)\s*(.+)$', 1, 'paren'),
            
            # Section keyword patterns: "1 Introduction" (no period)
            (r'^\s*(\d+)\s+(Introduction|Background|Methods|Methodology|Approach|Results|Discussion|Conclusion|References|Bibliography|Appendix)\s*$', 1, 'keyword'),
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            for pattern, level, section_type in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    # Extract section number and title
                    if section_type in ['arabic', 'roman', 'letter', 'paren']:
                        number = groups[0]
                        title = groups[1] if len(groups) > 1 else ''
                    elif section_type == 'arabic_sub':
                        number = groups[0]
                        title = groups[1] if len(groups) > 1 else ''
                    elif section_type == 'uppercase':
                        number = ''
                        title = groups[0] if groups else ''
                    elif section_type == 'keyword':
                        number = groups[0]
                        title = groups[1] if len(groups) > 1 else ''
                    
                    # Clean title
                    title = title.strip()
                    if len(title) > 100:
                        title = title[:100]  # Truncate very long titles
                    
                    # Determine if this is a special section
                    is_special = self._is_special_section(title)
                    
                    section = {
                        'line': i,
                        'level': level,
                        'number': number,
                        'title': title,
                        'type': section_type,
                        'content_start': i + 1,
                        'is_special': is_special,
                        'content': '',  # Will be filled later
                        'subsections': []
                    }
                    
                    sections.append(section)
                    break
        
        # Extract content for each section
        sections = self._extract_section_content(lines, sections)
        
        return sections
    
    def _is_special_section(self, title: str) -> str:
        """
        Identify special section types from title.
        """
        title_lower = title.lower()
        
        special_sections = {
            'references': ['reference', 'bibliography', 'literature cited'],
            'appendix': ['appendix', 'appendices', 'supplementary'],
            'acknowledgment': ['acknowledg', 'thank'],
            'abstract': ['abstract'],
            'keywords': ['keyword', 'subject'],
            'introduction': ['introduction'],
            'conclusion': ['conclusion', 'conclusions'],
            'methodology': ['method', 'methodology', 'approach'],
            'results': ['result', 'finding', 'evaluation'],
            'discussion': ['discussion'],
            'related_work': ['related work', 'background', 'prior work'],
        }
        
        for section_type, keywords in special_sections.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return section_type
        
        return ''
    
    def _extract_section_content(self, lines: List[str], sections: List[Dict]) -> List[Dict]:
        """
        Extract the content for each section.
        
        Determines section boundaries based on next section or document end.
        """
        if not sections:
            return sections
        
        # Build section hierarchy (parent-child relationships)
        for i, section in enumerate(sections):
            # Find end of section (next section or end of document)
            if i < len(sections) - 1:
                section['content_end'] = sections[i + 1]['line']
            else:
                section['content_end'] = len(lines)
            
            # Extract content lines
            content_lines = lines[section['content_start']:section['content_end']]
            section['content'] = '\n'.join(content_lines).strip()
            
            # Count subsections, figures, tables, equations
            section['subsection_count'] = 0
            section['figure_count'] = section['content'].count('Figure') + section['content'].count('Fig.')
            section['table_count'] = section['content'].count('Table')
            section['equation_count'] = len(re.findall(r'\((\d+(?:\.\d+)?)\)', section['content']))
            
            # Estimate reading time (avg 200 words per minute)
            word_count = len(section['content'].split())
            section['word_count'] = word_count
            section['reading_time_minutes'] = max(1, word_count // 200)
        
        return sections
    
    def _build_section_outline(self, sections: List[Dict]) -> Dict:
        """
        Build a hierarchical outline from flat section list.
        
        Returns:
            Nested dictionary with section hierarchy
        """
        if not sections:
            return {'children': [], 'depth': 0}
        
        # Build tree structure
        root = {
            'title': 'Document',
            'level': -1,
            'number': '',
            'children': [],
            'sections_at_level': {i: [] for i in range(5)}
        }
        
        # Group sections by level
        level_groups = {}
        for section in sections:
            level = section.get('level', 0)
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(section)
        
        # Build hierarchy starting from level 0 (or lowest level if no level 0)
        min_level = min(level_groups.keys()) if level_groups else 0
        
        # Find top-level sections
        for section in level_groups.get(min_level, []):
            node = self._section_to_node(section, level_groups)
            root['children'].append(node)
        
        # Count total sections at each level
        for level, level_sections in level_groups.items():
            root['sections_at_level'][level] = len(level_sections)
        
        root['total_sections'] = len(sections)
        root['depth'] = max(level_groups.keys()) if level_groups else 0
        
        return root
    
    def _section_to_node(self, section: Dict, level_groups: Dict) -> Dict:
        """
        Convert section dict to tree node with children.
        """
        node = {
            'title': section.get('title', ''),
            'number': section.get('number', ''),
            'level': section.get('level', 0),
            'type': section.get('type', ''),
            'is_special': section.get('is_special', ''),
            'word_count': section.get('word_count', 0),
            'reading_time_minutes': section.get('reading_time_minutes', 1),
            'children': [],
            'has_subsections': False
        }
        
        # Find direct children (sections at next level)
        current_level = section.get('level', 0)
        next_level = current_level + 1
        
        for next_section in level_groups.get(next_level, []):
            # Check if this section is a child of current section
            if self._is_descendant(section, next_section, level_groups):
                child_node = self._section_to_node(next_section, level_groups)
                node['children'].append(child_node)
                node['has_subsections'] = True
        
        return node
    
    def _is_descendant(self, parent: Dict, potential_child: Dict, level_groups: Dict) -> bool:
        """
        Determine if potential_child is a descendant of parent.
        
        Uses position and level information.
        """
        parent_line = parent.get('line', 0)
        child_line = potential_child.get('line', 0)
        
        # Child must come after parent
        if child_line <= parent_line:
            return False
        
        # Check if there's an intermediate section at parent's level
        # between parent and child
        parent_level = parent.get('level', 0)
        
        for section in level_groups.get(parent_level, []):
            section_line = section.get('line', 0)
            if parent_line < section_line < child_line:
                # There's another section at parent's level between them
                # So this is not a direct child
                if section.get('number', '') != parent.get('number', ''):
                    return False
        
        return True
    
    def _get_section_by_number(self, outline: Dict, section_number: str) -> Optional[Dict]:
        """
        Find a section in the outline by its number.
        
        Args:
            outline: Hierarchical outline
            section_number: Section number (e.g., '1', '1.1', 'A')
            
        Returns:
            Section node or None if not found
        """
        def search_node(node):
            if node.get('number') == section_number:
                return node
            for child in node.get('children', []):
                result = search_node(child)
                if result:
                    return result
            return None
        
        return search_node(outline)
    
    def _get_section_hierarchy(self, outline: Dict, target_line: int) -> List[Dict]:
        """
        Get the section hierarchy containing a specific line number.
        
        Args:
            outline: Hierarchical outline
            target_line: Line number to find
            
        Returns:
            List of sections from root to leaf containing the line
        """
        def search_node(node, path):
            # Check if this section contains the target line
            # This requires section content information
            section_info = {
                'title': node.get('title', ''),
                'number': node.get('number', ''),
                'level': node.get('level', 0),
                'path': path + [node]
            }
            
            # Check children
            for child in node.get('children', []):
                result = search_node(child, path + [node])
                if result:
                    return result
            
            return None
        
        return search_node(outline, [])
    
    def _format_outline_markdown(self, outline: Dict, max_depth: int = 3, 
                                 include_content: bool = False) -> str:
        """
        Format the hierarchical outline as Markdown.
        
        Args:
            outline: Hierarchical outline dict
            max_depth: Maximum depth to display
            include_content: Include section content summaries
            
        Returns:
            Markdown formatted outline
        """
        if not outline:
            return ''
        
        md_lines = ['\n## Document Outline\n']
        
        def format_node(node, depth: int, prefix: str = ''):
            if depth > max_depth:
                return
            
            title = node.get('title', '')
            number = node.get('number', '')
            level = node.get('level', -1)
            
            # Skip root node
            if level < 0:
                for child in node.get('children', []):
                    format_node(child, depth, '')
                return
            
            # Format section entry
            if number:
                entry = f"{prefix}{number} {title}"
            else:
                entry = f"{prefix}{title}"
            
            indent = '  ' * depth
            md_lines.append(f"{indent}- {entry}")
            
            # Add metadata
            if include_content:
                word_count = node.get('word_count', 0)
                reading_time = node.get('reading_time_minutes', 0)
                if word_count > 0:
                    md_lines.append(f"{indent}  *{word_count} words, ~{reading_time} min*")
            
            # Process children
            for child in node.get('children', []):
                format_node(child, depth + 1, prefix)
        
        format_node(outline, 0)
        
        # Add summary
        md_lines.append('')
        md_lines.append('### Summary')
        md_lines.append(f"- Total sections: {outline.get('total_sections', 0)}")
        md_lines.append(f"- Outline depth: {outline.get('depth', 0)}")
        
        for level, count in outline.get('sections_at_level', {}).items():
            if count > 0:
                md_lines.append(f"- Level {level} sections: {count}")
        
        return '\n'.join(md_lines)
    
    def _extract_section_text(self, text: str, section_number: str) -> str:
        """
        Extract the full text of a specific section.
        
        Args:
            text: Full document text
            section_number: Number of section to extract
            
        Returns:
            Section text or empty string if not found
        """
        # This is a simplified implementation
        # A more robust version would use the outline structure
        
        import re
        
        # Find section header
        patterns = [
            rf'^\s*{re.escape(section_number)}\.\s+(.+)$',
            rf'^\s*{re.escape(section_number)}\s+(.+)$',
        ]
        
        section_start = -1
        section_title = ''
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                section_start = match.start()
                section_title = match.group(1).strip()
                break
        
        if section_start < 0:
            return ''
        
        # Find end of section (next section or end)
        lines = text[section_start:].split('\n')
        
        end_markers = [
            r'^\s*\d+\.\s+',  # Next numbered section
            r'^\s*[A-Z][A-Z\s]+\s*$',  # ALL CAPS header
            r'^\s*References?\s*$',
            r'^\s*Appendix',
        ]
        
        section_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                section_lines.append(line)
                continue
            
            # Check if this line starts a new section
            is_new_section = any(re.match(m, line) for m in end_markers)
            
            if is_new_section:
                break
            
            section_lines.append(line)
        
        return '\n'.join(section_lines)
    
    def _validate_section_structure(self, sections: List[Dict]) -> Dict:
        """
        Validate the detected section structure for common issues.
        
        Returns:
            Dict with validation results and suggestions
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'statistics': {}
        }
        
        if not sections:
            validation['is_valid'] = False
            validation['issues'].append('No sections detected')
            return validation
        
        # Check for numbering gaps
        numbers = []
        for section in sections:
            num = section.get('number', '')
            if num:
                numbers.append(num)
        
        # Check for expected sections
        section_titles = [s.get('title', '').lower() for s in sections]
        
        expected_sections = {
            'introduction': any('intro' in t for t in section_titles),
            'conclusion': any('conclusion' in t for t in section_titles),
            'references': any('reference' in t for t in section_titles),
        }
        
        validation['expected_sections'] = expected_sections
        
        # Check section hierarchy
        levels = [s.get('level', 0) for s in sections]
        if levels:
            max_level = max(levels)
            validation['statistics']['max_depth'] = max_level
            validation['statistics']['total_sections'] = len(sections)
            
            # Check for level jumps (e.g., level 0 to level 2)
            for i in range(len(levels) - 1):
                if levels[i + 1] > levels[i] + 1:
                    validation['issues'].append(
                        f"Level jump detected at section {i + 2}: "
                        f"jumped from level {levels[i]} to {levels[i + 1]}"
                    )
                    validation['is_valid'] = False
        
        # Suggest improvements
        if not expected_sections['introduction']:
            validation['suggestions'].append(
                "Consider adding an 'Introduction' section if missing"
            )
        if not expected_sections['conclusion']:
            validation['suggestions'].append(
                "Consider adding a 'Conclusion' section if missing"
            )
        
        return validation
    
    def _format_abstract_markdown(self, abstract_info: Dict) -> str:
        """
        Format abstract information as Markdown.
        """
        if not abstract_info or not abstract_info.get('has_abstract'):
            return ''
        
        md_lines = ['\n## Abstract\n']
        md_lines.append(abstract_info.get('abstract', ''))
        
        # Add confidence indicator
        confidence = abstract_info.get('confidence', 0)
        if confidence >= 0.8:
            confidence_text = 'High confidence'
        elif confidence >= 0.5:
            confidence_text = 'Medium confidence'
        else:
            confidence_text = 'Low confidence'
        
        md_lines.append(f'\n*Extraction confidence: {confidence_text} ({confidence:.2f})*')
        
        return '\n'.join(md_lines)
    
    def _format_structure_markdown(self, structure: Dict) -> str:
        """
        Format document structure as Markdown outline.
        """
        if not structure:
            return ''
        
        md_lines = ['\n## Document Structure\n']
        
        # Abstract
        if structure.get('abstract'):
            md_lines.append('### Abstract')
            md_lines.append(structure['abstract'].get('abstract', '')[:500])
            md_lines.append('')
        
        # Sections
        if structure.get('sections'):
            md_lines.append('### Sections')
            for section in structure['sections'][:20]:  # Limit to first 20
                indent = '  ' * section.get('level', 0)
                number = section.get('number', '')
                if number:
                    md_lines.append(f"{indent}- {number} {section.get('title', '')}")
                else:
                    md_lines.append(f"{indent}- {section.get('title', '')}")
            md_lines.append('')
        
        # Keywords
        if structure.get('keywords'):
            md_lines.append('### Keywords')
            md_lines.append(', '.join(structure['keywords']))
            md_lines.append('')
        
        # Statistics
        md_lines.append('### Statistics')
        md_lines.append(f"- Sections: {len(structure.get('sections', []))}")
        md_lines.append(f"- Figures: {len(structure.get('figures', []))}")
        md_lines.append(f"- Tables: {len(structure.get('tables', []))}")
        md_lines.append(f"- Equations: {len(structure.get('equations', []))}")
        
        return '\n'.join(md_lines)
    
    def _validate_extraction(self, extracted: Dict, min_length: int = 100) -> bool:
        """Validate that extraction produced meaningful text with better edge case handling."""
        text = extracted.get('text', '')
        pages = extracted.get('pages', [])
        page_count = len(pages) if pages else 1
        
        # For very short papers, use lower minimum
        if page_count <= 5:
            min_length = max(50, min_length // 2)  # Half the minimum for very short papers
        
        if not text or len(text.strip()) < min_length:
            return False
        
        # Check if text is mostly whitespace or special characters
        non_whitespace = len(re.sub(r'\s+', '', text))
        if non_whitespace < min_length * 0.5:
            return False
        
        # Additional quality checks
        # Check for reasonable word count (not just symbols)
        words = text.split()
        if len(words) < min_length // 10:  # At least some words
            return False
        
        # For short papers, be more lenient
        if page_count <= 10:
            if len(words) < min_length // 20:  # More lenient word count
                return False
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:  # At least 2 sentences (more lenient)
            # For very short papers, allow single sentence if it's substantial
            if page_count <= 5 and len(text) > min_length * 2:
                pass  # Allow it
            else:
                return False
        
        # Check for reasonable word-to-character ratio (should be mostly words, not symbols)
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 1.5:  # Too many single characters (more lenient)
                return False
            # For short papers with tables/formulas, allow longer words
            if page_count <= 10 and avg_word_length > 20:
                # Might have technical terms or formulas, check if reasonable
                long_words = sum(1 for w in words if len(w) > 15)
                if long_words > len(words) * 0.3:  # More than 30% very long words
                    return False
        
        # Edge case: Check for corrupted or mostly empty extraction
        # If most pages are empty, likely an issue
        if pages:
            empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 50)
            if empty_pages > page_count * 0.5:  # More than 50% empty pages
                return False
        
        return True
    
    def _post_process_extracted_text(self, text: str) -> str:
        """
        Post-process extracted text to improve quality.
        Fixes common extraction issues.

        Uses pre-compiled regex patterns for better performance.
        Improved line break handling for better text flow.
        """
        if not text:
            return text

        # Apply all pre-compiled simple patterns
        for pattern, replacement in _POST_PATTERNS_SIMPLE:
            text = pattern.sub(replacement, text)

        # Apply subscript/superscript patterns (must run after simple patterns)
        for pattern, replacement in _POST_PATTERNS_SUBSUP:
            text = pattern.sub(replacement, text)

        # Fix broken multi-column text (lambda function)
        text = _POST_MULTICOLUMN_PATTERN.sub(
            lambda m: m.group(1) + ' ' + m.group(2) + ' ' + m.group(3)
            if len(m.group(2)) == 1 and m.group(1)[-1].islower() and m.group(3)[0].islower()
            else m.group(0),
            text
        )

        # Note: Quote normalization is now handled in _fix_encoding_issues
        # This section previously contained manual quote replacement

        # Improved line processing with better paragraph detection
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Strip whitespace
            stripped_line = line.strip()
            
            # Skip empty lines for now (handle later)
            if not stripped_line:
                continue
            
            # Handle single character lines intelligently
            if len(stripped_line) == 1:
                # Check if this is likely an initial or acronym (uppercase with periods)
                if stripped_line.isupper() or (stripped_line.isalpha() and i > 0):
                    # Check previous line to see if this should be attached
                    if processed_lines:
                        prev_line = processed_lines[-1]
                        # If previous line ends with a period, this might be an initial
                        if prev_line.rstrip().endswith('.'):
                            processed_lines[-1] = prev_line + ' ' + stripped_line
                            continue
                        # If previous line is short and this is uppercase, might be start of new para
                        elif len(prev_line) < 50 and prev_line[0].isupper():
                            processed_lines.append(stripped_line)
                            continue
                
                processed_lines.append(stripped_line)
            
            # Handle very short lines (2-4 chars)
            elif len(stripped_line) <= 4:
                if processed_lines:
                    prev_line = processed_lines[-1]
                    # Check if this should join with previous (broken word)
                    if prev_line and prev_line[-1].islower() and stripped_line[0].islower():
                        # Likely a broken word - join them
                        processed_lines[-1] = prev_line + stripped_line
                        continue
                    # Check if this should join with next (will check next iteration)
                    processed_lines.append(stripped_line)
                else:
                    processed_lines.append(stripped_line)
            
            # Normal length line
            else:
                processed_lines.append(stripped_line)
        
        # Now handle paragraph breaks
        # Group lines into paragraphs and clean up
        paragraphs = []
        current_para = []
        
        for line in processed_lines:
            # Check if this line should start a new paragraph
            # Heuristics:
            # 1. Line is very short (likely a header or单独一行)
            # 2. Line starts with uppercase and previous line ended with sentence punctuation
            # 3. Line is much shorter than average and starts with uppercase
            
            if current_para:
                prev_line = current_para[-1]
                # If previous line ends with sentence-ending punctuation
                # and this line starts with uppercase, it might be a new sentence or new paragraph
                if prev_line[-1] in '.!?' and line and line[0].isupper():
                    # Check if this looks like a new paragraph (very short line)
                    if len(line) < 30:
                        # Check if previous line is also short - might be headers
                        if len(prev_line) < 30:
                            # Likely headers/captions - keep separate
                            paragraphs.append(' '.join(current_para))
                            current_para = [line]
                        else:
                            # Could be new paragraph or continuation
                            # Check average line length
                            avg_len = sum(len(l) for l in current_para) / len(current_para)
                            if len(line) < avg_len * 0.5:
                                # Line is much shorter - likely new paragraph
                                paragraphs.append(' '.join(current_para))
                                current_para = [line]
                            else:
                                # Likely continuation - add to current paragraph
                                current_para.append(line)
                    else:
                        # Normal length line - likely continuation
                        current_para.append(line)
                else:
                    # Not sentence-ending - definitely continuation
                    current_para.append(line)
            else:
                # First line of a paragraph
                current_para.append(line)
        
        # Don't forget the last paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Join paragraphs with double newlines
        text = '\n\n'.join(paragraphs)
        
        # Final cleanup: normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text
    
    def _score_extraction_quality(self, extracted: Dict) -> float:
        """
        Score extraction quality (0-1) with comprehensive metrics.
        Higher score = better quality.
        """
        text = extracted.get('text', '')
        pages = extracted.get('pages', [])
        
        if not text or not pages:
            return 0.0
        
        # Base score components
        length_score = 1.0
        page_coverage_score = 1.0
        structure_score = 0.0
        readability_score = 1.0
        
        # 1. Length-based scoring (more nuanced, better handling of short papers)
        text_length = len(text)
        page_count = len(pages)
        
        # Expected text length: ~2000-3000 chars per page for academic papers
        # But be more lenient for short papers (some papers are legitimately short)
        expected_length = page_count * 2500
        
        if text_length < 500:
            length_score = 0.2
        elif text_length < 1000:
            length_score = 0.4
        elif text_length < expected_length * 0.3:
            # For very short papers, check if it's complete (has structure)
            # If it has good structure, be more lenient
            if page_count <= 10:  # Short paper
                length_score = 0.75  # More lenient for short but complete papers
            else:
                length_score = 0.6
        elif text_length < expected_length * 0.6:
            if page_count <= 10:  # Short paper
                length_score = 0.9  # Very lenient for short papers
            else:
                length_score = 0.8
        elif text_length < expected_length * 0.9:
            length_score = 0.9
        # If text is too long, might have duplicates (penalize slightly)
        elif text_length > expected_length * 2.0:
            length_score = 0.95
        
        # 2. Page coverage scoring
        empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 50)
        if empty_pages > 0:
            page_coverage_score = 1.0 - (empty_pages / page_count) * 0.3
        
        # Average characters per page
        if pages:
            avg_chars = sum(p.get('char_count', 0) for p in pages) / len(pages)
            # Be more lenient for short papers (might have more figures/tables/whitespace)
            if page_count <= 10:  # Short paper
                if avg_chars < 200:
                    page_coverage_score *= 0.6  # Less penalty
                elif avg_chars < 500:
                    page_coverage_score *= 0.8  # Less penalty
                elif avg_chars < 1000:
                    page_coverage_score *= 0.9  # Less penalty
                elif avg_chars < 1500:
                    page_coverage_score *= 0.95  # Very lenient
            else:  # Longer papers
                if avg_chars < 200:
                    page_coverage_score *= 0.5
                elif avg_chars < 500:
                    page_coverage_score *= 0.7
                elif avg_chars < 1000:
                    page_coverage_score *= 0.85
            # For papers with many pages, be more lenient (might have figures/tables)
            if page_count > 20 and avg_chars > 500:
                page_coverage_score = min(1.0, page_coverage_score * 1.1)
        
        # 3. Structure scoring (academic paper indicators) - improved detection
        text_lower = text.lower()
        
        # Improved abstract detection with fuzzy matching and alternatives
        abstract_patterns = [
            r'\babstract\b',
            r'\bsummary\b',
            r'\boverview\b',
            r'\bexecutive\s+summary\b',
            r'^abstract\s*:',  # At start of text
        ]
        has_abstract = any(re.search(pattern, text_lower[:3000], re.IGNORECASE) for pattern in abstract_patterns)
        
        # Improved introduction detection
        intro_patterns = [
            r'\bintroduction\b',
            r'\bintro\b',
            r'\bbackground\b',
            r'^introduction\s*:',  # At start
        ]
        has_introduction = any(re.search(pattern, text_lower[:5000], re.IGNORECASE) for pattern in intro_patterns)
        
        structure_indicators = {
            'abstract': has_abstract,
            'introduction': has_introduction,
            'references': any(term in text_lower for term in ['reference', 'bibliography', 'references', 'works cited', 'literature']),
            'methods': any(term in text_lower for term in ['method', 'methodology', 'approach', 'technique', 'procedure']),
            'results': any(term in text_lower for term in ['result', 'experiment', 'evaluation', 'analysis', 'findings']),
            'conclusion': any(term in text_lower for term in ['conclusion', 'conclusions', 'summary', 'discussion']),
            'figures': any(term in text_lower for term in ['figure', 'fig.', 'fig ', 'figures']),
            'tables': any(term in text_lower for term in ['table', 'tab.', 'tab ', 'tables']),
        }
        
        structure_count = sum(structure_indicators.values())
        # For short papers, be more lenient - they might not have all sections
        # Check if it has at least abstract (most important)
        if page_count <= 10 and structure_indicators.get('abstract', False):
            # Short paper with abstract - normalize to fewer required indicators
            structure_score = min(1.0, structure_count / 4.0)  # Only need 4 indicators
        else:
            structure_score = min(1.0, structure_count / 6.0)  # Normalize to 6 key indicators
        
        # 4. Readability scoring (check for common extraction issues)
        # Check for excessive whitespace (indicates poor extraction)
        whitespace_ratio = len(re.findall(r'\s+', text)) / max(len(text), 1)
        if whitespace_ratio > 0.3:  # More than 30% whitespace
            readability_score -= 0.2
        
        # Check for broken words (common extraction issue)
        # Pattern: single lowercase letter followed by space and uppercase (likely broken word)
        broken_words = len(re.findall(r'\b[a-z]\s+[A-Z][a-z]', text))
        if broken_words > len(text) / 1000:  # More than 1 per 1000 chars
            readability_score -= 0.15
        
        # Check for reasonable word-to-character ratio
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 2.0:  # Too many single characters
                readability_score -= 0.3
            elif avg_word_length > 15:  # Suspiciously long words (might be broken)
                readability_score -= 0.1
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+\s+', text)
        if len(sentences) < 5:
            readability_score -= 0.1
        
        # Check for common academic terms (positive indicator)
        academic_terms = ['algorithm', 'theorem', 'proof', 'lemma', 'corollary', 
                         'proposition', 'definition', 'hypothesis', 'dataset', 'baseline']
        academic_term_count = sum(1 for term in academic_terms if term in text_lower)
        if academic_term_count >= 3:
            readability_score += 0.1  # Bonus for academic content
        
        readability_score = max(0.0, min(1.0, readability_score))
        
        # 5. Combine scores with weights
        # For short texts/papers, be more lenient and weight structure/readability more
        if text_length < 5000 or page_count <= 10:
            # Short papers: weight structure and readability more, length less
            final_score = (
                length_score * 0.20 +  # Less weight on length
                page_coverage_score * 0.20 +  # Less weight on coverage
                structure_score * 0.35 +  # More weight on structure
                readability_score * 0.25  # More weight on readability
            )
        else:
            final_score = (
                length_score * 0.30 +
                page_coverage_score * 0.30 +
                structure_score * 0.20 +
                readability_score * 0.20
            )
        
        # Bonus for short papers with good structure and readability
        if page_count <= 10 and structure_score >= 0.7 and readability_score >= 0.8:
            final_score = min(1.0, final_score + 0.05)  # Small bonus
        
        return max(0.0, min(1.0, final_score))

    # ============================================================
    # Enhanced Quality Scoring
    # ============================================================

    def _score_extraction_quality_enhanced(self, extracted: Dict) -> Dict:
        """
        Multi-dimensional quality scoring for extraction results.

        Scores across multiple dimensions:
        - Completeness: How much content was captured
        - Coherence: Text flow and readability
        - Structure: Document structure preservation
        - Accuracy: Character/word accuracy
        - Metadata: Quality of extracted metadata
        - Consistency: Page-to-page consistency

        Args:
            extracted: Extraction result dictionary

        Returns:
            Dict with:
                - 'overall': Overall quality score (0-1)
                - 'dimensions': Dict of per-dimension scores
                - 'issues': List of identified quality issues
                - 'recommendations': List of improvement recommendations
                - 'confidence': Confidence level in the scores
        """
        import re

        text = extracted.get('text', '')
        pages = extracted.get('pages', [])
        metadata = extracted.get('metadata', {})

        result = {
            'overall': 0.0,
            'dimensions': {},
            'issues': [],
            'recommendations': [],
            'confidence': 0.0,
            'metadata': {
                'text_length': len(text),
                'page_count': len(pages),
                'word_count': len(text.split()),
                'sentence_count': 0,
                'avg_chars_per_page': 0.0
            }
        }

        if not text:
            result['issues'].append('No text extracted')
            result['recommendations'].append('Check PDF is text-based, not scanned')
            return result

        # Update metadata
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        result['metadata']['sentence_count'] = len(sentences)

        if pages:
            total_chars = sum(p.get('char_count', 0) for p in pages)
            result['metadata']['avg_chars_per_page'] = total_chars / len(pages)

        # 1. Completeness Score
        completeness_score = self._score_completeness(extracted)
        result['dimensions']['completeness'] = completeness_score

        # 2. Coherence Score
        coherence_score = self._score_text_coherence(text)
        result['dimensions']['coherence'] = coherence_score

        # 3. Structure Score
        structure_score = self._score_structure_preservation(extracted)
        result['dimensions']['structure'] = structure_score

        # 4. Accuracy Score
        accuracy_score = self._score_extraction_accuracy(text)
        result['dimensions']['accuracy'] = accuracy_score

        # 5. Metadata Score
        metadata_score = self._score_metadata_completeness(metadata)
        result['dimensions']['metadata'] = metadata_score

        # 6. Consistency Score
        consistency_score = self._score_page_consistency(pages)
        result['dimensions']['consistency'] = consistency_score

        # Calculate overall score with weighted dimensions
        weights = {
            'completeness': 0.25,
            'coherence': 0.20,
            'structure': 0.20,
            'accuracy': 0.15,
            'metadata': 0.10,
            'consistency': 0.10
        }

        result['overall'] = sum(
            result['dimensions'].get(dim, 0) * weight
            for dim, weight in weights.items()
        )

        # Identify specific issues
        result['issues'] = self._identify_quality_issues(result['dimensions'], result['metadata'])

        # Generate recommendations based on issues
        result['recommendations'] = self._generate_recommendations(result['issues'], result['dimensions'])

        # Calculate confidence based on amount of data
        result['confidence'] = self._calculate_scoring_confidence(result['metadata'])

        return result

    def _score_completeness(self, extracted: Dict) -> float:
        """Score how complete the extraction is."""
        text = extracted.get('text', '')
        pages = extracted.get('pages', [])

        if not text or not pages:
            return 0.0

        page_count = len(pages)
        total_chars = sum(p.get('char_count', 0) for p in pages)

        # Expected characters per page varies by paper type
        # Short papers might have more figures/tables, fewer characters
        empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 100)

        # Completeness factors
        # 1. Page coverage
        page_coverage = 1.0 - (empty_pages / max(page_count, 1)) * 0.5

        # 2. Character density vs expected
        expected_chars = page_count * 2500  # ~2500 chars per page average
        if expected_chars > 0:
            density = min(1.0, total_chars / expected_chars)
            if total_chars < expected_chars * 0.3:
                density *= 0.8  # Penalty for very low density
            elif total_chars < expected_chars * 0.6:
                density *= 0.9  # Minor penalty
        else:
            density = 0.5

        # 3. Section coverage
        text_lower = text.lower()
        required_sections = ['abstract', 'introduction']
        optional_sections = ['method', 'methodology', 'approach',
                           'result', 'experiment', 'evaluation',
                           'conclusion', 'discussion',
                           'reference', 'bibliography']

        has_required = sum(1 for s in required_sections if s in text_lower[:5000])
        has_optional = sum(1 for s in optional_sections if s in text_lower)

        section_score = 0.0
        if has_required == len(required_sections):
            section_score = 0.7  # Base score for having all required sections
            section_score += min(0.3, has_optional / 5.0)  # Bonus for optional sections
        else:
            section_score = 0.5 * (has_required / len(required_sections))

        # Combine scores
        completeness = (page_coverage * 0.4 + density * 0.3 + section_score * 0.3)

        return max(0.0, min(1.0, completeness))

    def _score_text_coherence(self, text: str) -> float:
        """Score text coherence based on sentence/paragraph structure."""
        import re

        if not text:
            return 0.0

        coherence_scores = []

        # 1. Sentence structure
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

            # Good sentence length is between 50-200 characters
            if 50 <= avg_sentence_length <= 200:
                sentence_score = 1.0
            elif avg_sentence_length < 50:
                sentence_score = 0.6  # Too short, might be fragments
            else:
                sentence_score = 0.7  # Long sentences might be run-on
        else:
            sentence_score = 0.0

        coherence_scores.append(sentence_score * 0.3)

        # 2. Broken words check
        # Pattern: lowercase letter followed by space and uppercase (broken word)
        broken_words = len(re.findall(r'\b[a-z]\s+[A-Z][a-z]', text))
        word_count = max(len(text.split()), 1)
        broken_ratio = broken_words / word_count

        if broken_ratio < 0.001:
            broken_score = 1.0
        elif broken_ratio < 0.01:
            broken_score = 0.8
        elif broken_ratio < 0.05:
            broken_score = 0.5
        else:
            broken_score = 0.2

        coherence_scores.append(broken_score * 0.25)

        # 3. Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            para_lengths = [len(p) for p in paragraphs]
            avg_para_length = sum(para_lengths) / len(para_lengths)

            # Good paragraph length is 200-1000 characters
            if 200 <= avg_para_length <= 1000:
                para_score = 1.0
            elif avg_para_length < 200:
                para_score = 0.6  # Too fragmented
            else:
                para_score = 0.7  # Very long paragraphs
        else:
            para_score = 0.5  # No clear paragraph breaks

        coherence_scores.append(para_score * 0.25)

        # 4. Word variety
        words = text.split()
        if words:
            unique_words = set(w.lower() for w in words)
            word_variety = len(unique_words) / len(words)

            # Good variety indicates natural text
            if word_variety > 0.3:
                variety_score = 1.0
            elif word_variety > 0.2:
                variety_score = 0.8
            else:
                variety_score = 0.5  # Might be repetitive or tables
        else:
            variety_score = 0.0

        coherence_scores.append(variety_score * 0.2)

        return max(0.0, min(1.0, sum(coherence_scores)))

    def _score_structure_preservation(self, extracted: Dict) -> float:
        """Score how well document structure is preserved."""
        text = extracted.get('text', '')
        text_lower = text.lower()

        if not text:
            return 0.0

        structure_score = 0.0

        # Check for academic paper structure
        structure_markers = {
            'abstract': ['abstract', 'summary', 'synopsis'],
            'introduction': ['introduction', 'intro', 'background', 'motivation'],
            'methodology': ['method', 'methodology', 'approach', 'technique', 'procedure'],
            'results': ['result', 'experiment', 'evaluation', 'analysis', 'finding'],
            'discussion': ['discussion', 'interpretation', 'implication'],
            'conclusion': ['conclusion', 'conclusions', 'future work', 'limitations'],
            'references': ['reference', 'bibliography', 'bibliographic', 'citations']
        }

        found_markers = 0
        total_markers = 0

        # Check each section type
        for section, markers in structure_markers.items():
            total_markers += 1
            if any(marker in text_lower for marker in markers):
                found_markers += 1

        # Check for numbered sections (indicates good structure extraction)
        numbered_sections = len(re.findall(r'\n\s*\d+\.\s+[A-Z]', text))
        section_pattern_score = min(1.0, numbered_sections / 5.0)  # Cap at 5 sections

        # Check for figure/table references (indicates structural elements preserved)
        fig_refs = len(re.findall(r'[Ff]ig(?:ure)?\.?\s*\d+', text))
        tab_refs = len(re.findall(r'[Tt]ab(?:le)?\.?\s*\d+', text))
        ref_score = min(1.0, (fig_refs + tab_refs) / 10.0)

        structure_score = (found_markers / total_markers) * 0.5 + \
                          section_pattern_score * 0.25 + \
                          ref_score * 0.25

        return max(0.0, min(1.0, structure_score))

    def _score_extraction_accuracy(self, text: str) -> float:
        """Score extraction accuracy based on common error patterns."""
        import re

        if not text:
            return 0.0

        accuracy_scores = []

        # 1. Check for excessive special characters (indicates OCR errors)
        special_chars = len(re.findall(r'[^\w\s.,!?;:\'"-]', text))
        special_ratio = special_chars / max(len(text), 1)

        if special_ratio < 0.02:
            special_score = 1.0
        elif special_ratio < 0.05:
            special_score = 0.8
        elif special_ratio < 0.10:
            special_score = 0.5
        else:
            special_score = 0.2

        accuracy_scores.append(special_score * 0.25)

        # 2. Check for common encoding issues
        encoding_issues = text.count('\ufffd')  # Unicode replacement character
        encoding_ratio = encoding_issues / max(len(text), 1)

        if encoding_ratio < 0.001:
            encoding_score = 1.0
        elif encoding_ratio < 0.01:
            encoding_score = 0.7
        else:
            encoding_score = 0.3

        accuracy_scores.append(encoding_score * 0.25)

        # 3. Check for abnormal whitespace patterns
        consecutive_spaces = len(re.findall(r' {3,}', text))
        consecutive_newlines = len(re.findall(r'\n{4,}', text))
        whitespace_ratio = (consecutive_spaces + consecutive_newlines) / max(len(text) / 1000, 1)

        if whitespace_ratio < 1:
            whitespace_score = 1.0
        elif whitespace_ratio < 5:
            whitespace_score = 0.7
        else:
            whitespace_score = 0.4

        accuracy_scores.append(whitespace_score * 0.25)

        # 4. Check for common extraction artifacts
        # Hyphenated line breaks (words broken across lines)
        hyphen_breaks = len(re.findall(r'\w-\s*\n\s*\w', text))
        break_ratio = hyphen_breaks / max(len(text) / 5000, 1)  # Normalize

        if break_ratio < 0.5:
            break_score = 1.0
        elif break_ratio < 2:
            break_score = 0.7
        else:
            break_score = 0.4

        accuracy_scores.append(break_score * 0.25)

        return max(0.0, min(1.0, sum(accuracy_scores)))

    def _score_metadata_completeness(self, metadata: Dict) -> float:
        """Score quality of extracted metadata."""
        if not metadata:
            return 0.0

        metadata_fields = {
            'title': {'weight': 0.3, 'min_length': 5},
            'author': {'weight': 0.25, 'min_length': 3},
            'page_count': {'weight': 0.15, 'check': lambda v: v > 0},
            'keywords': {'weight': 0.15, 'check': lambda v: len(v) > 0 if isinstance(v, list) else len(str(v)) > 0},
            'subject': {'weight': 0.10, 'min_length': 5},
            'creator': {'weight': 0.05, 'min_length': 1}
        }

        score = 0.0
        total_weight = 0.0

        for field, config in metadata_fields.items():
            if field in metadata and metadata[field]:
                value = metadata[field]
                weight = config['weight']

                # Check if value meets minimum quality
                if 'min_length' in config:
                    if len(str(value).strip()) >= config['min_length']:
                        score += weight
                elif 'check' in config:
                    if config['check'](value):
                        score += weight

            total_weight += weight

        # Normalize to 0-1
        if total_weight > 0:
            score = score / total_weight

        return max(0.0, min(1.0, score))

    def _score_page_consistency(self, pages: List[Dict]) -> float:
        """Score consistency across pages."""
        if not pages or len(pages) <= 1:
            return 1.0  # Can't measure consistency with 0-1 pages

        char_counts = [p.get('char_count', 0) for p in pages]

        if not char_counts:
            return 0.5

        # Calculate coefficient of variation
        avg_chars = sum(char_counts) / len(char_counts)
        if avg_chars == 0:
            return 0.5

        variance = sum((c - avg_chars) ** 2 for c in char_counts) / len(char_counts)
        std_dev = variance ** 0.5
        cv = std_dev / avg_chars  # Coefficient of variation

        # Lower CV means more consistent (better)
        if cv < 0.3:
            consistency = 1.0
        elif cv < 0.5:
            consistency = 0.8
        elif cv < 0.8:
            consistency = 0.6
        else:
            consistency = 0.4

        # Check for empty pages (anomaly)
        empty_pages = sum(1 for c in char_counts if c < 100)
        if empty_pages > 0:
            empty_ratio = empty_pages / len(char_counts)
            consistency *= (1.0 - empty_ratio * 0.5)

        return max(0.0, min(1.0, consistency))

    def _identify_quality_issues(self, dimensions: Dict, metadata: Dict) -> List[str]:
        """Identify specific quality issues from dimension scores."""
        issues = []

        # Check each dimension
        if dimensions.get('completeness', 1.0) < 0.6:
            issues.append('Low content completeness')

        if dimensions.get('coherence', 1.0) < 0.5:
            issues.append('Text coherence issues detected')

        if dimensions.get('structure', 1.0) < 0.5:
            issues.append('Poor document structure preservation')

        if dimensions.get('accuracy', 1.0) < 0.6:
            issues.append('Extraction accuracy concerns')

        if dimensions.get('metadata', 1.0) < 0.5:
            issues.append('Incomplete metadata')

        if dimensions.get('consistency', 1.0) < 0.5:
            issues.append('Inconsistent page extraction')

        # Check specific metadata issues
        if metadata.get('avg_chars_per_page', 0) < 500:
            issues.append('Low characters per page (possible extraction issues)')

        return issues

    def _generate_recommendations(self, issues: List[str], dimensions: Dict) -> List[str]:
        """Generate improvement recommendations based on issues."""
        recommendations = []

        issue_mapping = {
            'completeness': [
                'Try using OCR for scanned documents',
                'Check if PDF has text layer',
                'Consider using a different extraction library'
            ],
            'coherence': [
                'Enable post-processing for better text flow',
                'Check for multi-column layout issues',
                'Consider using layout-preserving extraction'
            ],
            'structure': [
                'Document may not follow standard academic format',
                'Check if section headers are properly detected',
                'Consider manual structure verification'
            ],
            'accuracy': [
                'Enable encoding fix post-processing',
                'Check for PDF corruption',
                'Try extracting at higher resolution'
            ],
            'metadata': [
                'PDF metadata may be incomplete',
                'Consider extracting metadata from content',
                'Check if PDF has XMP metadata'
            ],
            'consistency': [
                'Some pages may have different layouts',
                'Check for mixed content types (text/images)',
                'Consider page-by-page quality check'
            ]
        }

        for issue in issues:
            if 'completeness' in issue.lower():
                recommendations.extend(issue_mapping['completeness'][:1])
            elif 'coherence' in issue.lower():
                recommendations.extend(issue_mapping['coherence'][:1])
            elif 'structure' in issue.lower():
                recommendations.extend(issue_mapping['structure'][:1])
            elif 'accuracy' in issue.lower():
                recommendations.extend(issue_mapping['accuracy'][:1])
            elif 'metadata' in issue.lower():
                recommendations.extend(issue_mapping['metadata'][:1])
            elif 'consistency' in issue.lower():
                recommendations.extend(issue_mapping['consistency'][:1])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)

        return unique_recommendations[:5]  # Limit to 5 recommendations

    def _calculate_scoring_confidence(self, metadata: Dict) -> float:
        """Calculate confidence in the quality scores based on data amount."""
        text_length = metadata.get('text_length', 0)
        page_count = metadata.get('page_count', 0)
        sentence_count = metadata.get('sentence_count', 0)

        confidence = 0.0

        # More text = more confident
        if text_length > 10000:
            confidence += 0.4
        elif text_length > 5000:
            confidence += 0.3
        elif text_length > 1000:
            confidence += 0.2
        else:
            confidence += 0.1

        # More pages = more confident
        if page_count > 20:
            confidence += 0.3
        elif page_count > 10:
            confidence += 0.2
        elif page_count > 5:
            confidence += 0.1

        # More sentences = more confident
        if sentence_count > 100:
            confidence += 0.3
        elif sentence_count > 50:
            confidence += 0.2
        elif sentence_count > 20:
            confidence += 0.1

        return min(1.0, confidence)

    def clean_text(self, text: str, remove_headers_footers: bool = True, 
                   preserve_math: bool = True) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            remove_headers_footers: Remove common headers/footers
            preserve_math: Preserve mathematical expressions in LaTeX format
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        if remove_headers_footers:
            # Use enhanced header/footer detection for better results
            text = self._remove_headers_footers_enhanced(text)
        
        if preserve_math:
            # Preserve mathematical expressions
            text = self._preserve_math_expressions(text)

        # Fix common encoding issues using comprehensive method
        text = self._fix_encoding_issues(text, preserve_math=preserve_math)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Fix special characters
        text = self._fix_special_characters(text)

        # Handle RTL text
        text = self._handle_rtl_text(text)

        return text.strip()
    
    def _preserve_math_expressions(self, text: str) -> str:
        """
        Detect and preserve mathematical expressions with proper formatting.
        
        This method:
        1. Detects inline math ($...$, \(...\))
        2. Detects display math ($$...$$, \[...\])
        3. Preserves equation structure and numbering
        4. Handles multi-line equations
        5. Restores equations after text processing
        """
        if not text:
            return text
        
        # Store original math expressions as placeholders
        inline_math = []
        display_math = []
        equation_numbers = []
        
        # Pattern 1: Display math with $$...$$ (may span multiple lines)
        def save_display_math_multiline(match):
            expr = match.group(0)
            placeholder = f"__MATH_DISPLAY_{len(display_math)}__"
            display_math.append(expr)
            return placeholder
        
        # Match $$...$$ with possible newlines inside
        text = re.sub(r'\$\$[^\$]*?\$\$', save_display_math_multiline, text, flags=re.DOTALL)
        
        # Pattern 2: Display math with \[...\]
        def save_display_bracket(match):
            expr = match.group(0)
            placeholder = f"__MATH_DISPLAY_{len(display_math)}__"
            display_math.append(expr)
            return placeholder
        
        text = re.sub(r'\\\[[^\]]*?\\\]', save_display_bracket, text, flags=re.DOTALL)
        
        # Pattern 3: Inline math with $...$ (be careful not to match $$)
        def save_inline_dollar(match):
            expr = match.group(0)
            # Make sure it's single $, not $$
            if expr.startswith('$$') or expr.endswith('$$'):
                return expr
            placeholder = f"__MATH_INLINE_{len(inline_math)}__"
            inline_math.append(expr)
            return placeholder
        
        # Match single $...$ but not $$...$$
        text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)', save_inline_dollar, text)
        
        # Pattern 4: Inline math with \(...\)
        def save_inline_bracket(match):
            expr = match.group(0)
            placeholder = f"__MATH_INLINE_{len(inline_math)}__"
            inline_math.append(expr)
            return placeholder
        
        text = re.sub(r'\\\((?:[^\\)]|\\.)*?\\\)', save_inline_bracket, text)
        
        # Pattern 5: Equation numbers like (1), (2.3), Eq. (5)
        def save_equation_number(match):
            eq = match.group(0)
            placeholder = f"__EQN_NUM_{len(equation_numbers)}__"
            equation_numbers.append(eq)
            return placeholder
        
        # Match various equation number formats
        eq_patterns = [
            r'\(\d+\)',                          # (1)
            r'\(\d+\.\d+\)',                     # (1.2)
            r'\(\d+\.\d+\.\d+\)',                # (1.2.3)
            r'[Ee]q\.?\s*\(\d+\)',               # Eq. (1)
            r'[Ee]quation\s*\(\d+\)',            # Equation (1)
            r'\[\d+\]',                          # [1]
        ]
        
        for pattern in eq_patterns:
            text = re.sub(pattern, save_equation_number, text)
        
        # Now text processing happens here (other post-processing)...
        # At this point, math expressions are protected as placeholders
        
        # Restore inline math expressions
        for i, expr in enumerate(inline_math):
            text = text.replace(f"__MATH_INLINE_{i}__", f" {expr} ")
        
        # Restore display math expressions with proper formatting
        for i, expr in enumerate(display_math):
            text = text.replace(f"__MATH_DISPLAY_{i}__", f"\n\n{expr}\n\n")
        
        # Restore equation numbers
        for i, eq in enumerate(equation_numbers):
            text = text.replace(f"__EQN_NUM_{i}__", eq)
        
        return text
    
    def _extract_math_expressions(self, text: str) -> Dict:
        """
        Extract mathematical expressions from text and return structured data.
        
        Returns:
            Dict with:
            - inline_math: list of inline expressions
            - display_math: list of display expressions
            - equation_numbers: list of equation references
            - cleaned_text: text with math replaced by placeholders
        """
        import re
        
        result = {
            'inline_math': [],
            'display_math': [],
            'equation_numbers': [],
            'cleaned_text': text
        }
        
        # Extract display math $$...$$
        def save_display(match):
            result['display_math'].append(match.group(0))
            return f"__MATH_DISPLAY_{len(result['display_math']) - 1}__"
        
        result['cleaned_text'] = re.sub(
            r'\$\$[^\$]*?\$\$', 
            save_display, 
            result['cleaned_text'], 
            flags=re.DOTALL
        )
        
        # Extract display math \[...\]
        result['cleaned_text'] = re.sub(
            r'\\\[[^\]]*?\\\]',
            save_display,
            result['cleaned_text'],
            flags=re.DOTALL
        )
        
        # Extract inline math $...$
        def save_inline(match):
            result['inline_math'].append(match.group(1))
            return f"__MATH_INLINE_{len(result['inline_math']) - 1}__"
        
        result['cleaned_text'] = re.sub(
            r'\$([^\$\n]+?)\$',
            save_inline,
            result['cleaned_text']
        )
        
        # Extract inline math \(...\)
        result['cleaned_text'] = re.sub(
            r'\\\((?:[^\\)]|\\.)*?\\\)',
            save_inline,
            result['cleaned_text']
        )
        
        # Extract equation numbers
        def save_eqn(match):
            result['equation_numbers'].append(match.group(0))
            return f"__EQN_NUM_{len(result['equation_numbers']) - 1}__"
        
        eq_patterns = [
            r'\(\d+\)',
            r'\(\d+\.\d+\)',
            r'\(\d+\.\d+\.\d+\)',
            r'[Ee]q\.?\s*\(\d+\)',
            r'[Ee]quation\s*\(\d+\)',
        ]
        
        for pattern in eq_patterns:
            result['cleaned_text'] = re.sub(pattern, save_eqn, result['cleaned_text'])
        
        return result
    
    def _restore_math_expressions(self, text: str, math_data: Dict) -> str:
        """
        Restore mathematical expressions from extracted data.
        
        Args:
            text: Text with placeholders
            math_data: Dict containing extracted math expressions
            
        Returns:
            Text with math expressions restored
        """
        # Restore inline math
        for i, expr in enumerate(math_data['inline_math']):
            text = text.replace(f"__MATH_INLINE_{i}__", f" ${expr} ")
        
        # Restore display math
        for i, expr in enumerate(math_data['display_math']):
            text = text.replace(f"__MATH_DISPLAY_{i}__", f"\n\n$${expr}$$\n\n")
        
        # Restore equation numbers
        for i, eq in enumerate(math_data['equation_numbers']):
            text = text.replace(f"__EQN_NUM_{i}__", eq)
        
        return text
    
    def _format_math_for_output(self, text: str, output_format: str = 'latex') -> str:
        """
        Format mathematical expressions for different output formats.
        
        Args:
            text: Text with math expressions
            output_format: 'latex', 'unicode', or 'text'
            
        Returns:
            Formatted text
        """
        if output_format == 'latex':
            # Keep LaTeX format as-is
            return text
        
        elif output_format == 'unicode':
            # Convert common LaTeX to Unicode symbols
            replacements = {
                r'\\alpha': 'α',
                r'\\beta': 'β',
                r'\\gamma': 'γ',
                r'\\delta': 'δ',
                r'\\epsilon': 'ε',
                r'\\theta': 'θ',
                r'\\lambda': 'λ',
                r'\\mu': 'μ',
                r'\\pi': 'π',
                r'\\sigma': 'σ',
                r'\\phi': 'φ',
                r'\\omega': 'ω',
                r'\\Delta': 'Δ',
                r'\\Omega': 'Ω',
                r'\\Sigma': 'Σ',
                r'\\times': '×',
                r'\\div': '÷',
                r'\\pm': '±',
                r'\\mp': '∓',
                r'\\leq': '≤',
                r'\\geq': '≥',
                r'\\neq': '≠',
                r'\\approx': '≈',
                r'\\equiv': '≡',
                r'\\infty': '∞',
                r'\\partial': '∂',
                r'\\nabla': '∇',
                r'\\sum': 'Σ',
                r'\\prod': 'Π',
                r'\\int': '∫',
                r'\\sqrt': '√',
                r'\\cdot': '·',
                r'\\leftarrow': '←',
                r'\\rightarrow': '→',
                r'\\Leftarrow': '⇐',
                r'\\Rightarrow': '⇒',
                r'\\leftrightarrow': '↔',
                r'\\Rightarrow': '⇒',
            }
            
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
            
            # Remove LaTeX command braces
            text = re.sub(r'\{([^}]+)\}', r'\1', text)
            text = re.sub(r'\\\s*', '', text)
            
            return text
        
        elif output_format == 'text':
            # Convert to plain text descriptions
            text = re.sub(r'\$([^\$]+?)\$', r' [math: \1] ', text)
            text = re.sub(r'\$\$([^\$]+?)\$\$', r'\n\n[displayed equation]\n\1\n[/displayed equation]\n\n', text, flags=re.DOTALL)
            text = re.sub(r'\\\(([^\\)]+?)\\\)', r' [math: \1] ', text)
            text = re.sub(r'\\\[([^\]]+?)\\\]', r'\n\n[displayed equation]\n\1\n[/displayed equation]\n\n', text, flags=re.DOTALL)
            
            return text
        
        else:
            return text
    
    def _detect_equations_in_blocks(self, blocks: List) -> List[Dict]:
        """
        Detect and extract equations from PDF blocks.
        
        This method identifies blocks that likely contain equations
        based on their content and formatting.
        """
        import re
        
        equations = []
        
        for i, block in enumerate(blocks):
            if len(block) < 5:
                continue
            
            text = block[4].strip() if len(block) > 4 and block[4] else ''
            if not text:
                continue
            
            # Check for equation-like patterns
            eq_patterns = [
                r'^[\d]+\.',                          # Numbered equation start
                r'^[\d]+\.[\d]+',                     # Sub-equation start
                r'^[\(\[][\d]+[\)\]]',                # (1) or [1] format
                r'.*=.*[\d\(\[]',                     # Contains = and equation ref
                r'\\begin\{.*\}',                     # LaTeX environment
                r'\\end\{.*\}',
                r'\\[a-zA-Z]+',                       # LaTeX command
                r'\^\{.*\}|\^.',                      # Superscript
                r'_\{.*\}|_.',                        # Subscript
            ]
            
            is_equation = any(re.search(p, text) for p in eq_patterns)
            
            # Check for standalone equation lines
            # (lines that are centered and contain math symbols)
            if is_equation:
                equations.append({
                    'block_index': i,
                    'text': text,
                    'y_position': block[1] if len(block) > 1 else 0,
                    'x_position': block[0] if len(block) > 0 else 0,
                })
        
        # Group consecutive equation blocks
        equation_groups = []
        current_group = []
        prev_y = None
        
        for eq in equations:
            if prev_y is not None and eq['y_position'] - prev_y > 30:
                # Significant gap - start new equation
                if current_group:
                    equation_groups.append(current_group)
                current_group = [eq]
            else:
                current_group.append(eq)
            prev_y = eq['y_position']
        
        if current_group:
            equation_groups.append(current_group)
        
        return equation_groups
    
    def _remove_headers_footers_enhanced(self, text: str) -> str:
        """
        Enhanced header/footer removal using content analysis.
        
        This method identifies headers and footers by:
        1. Finding content that appears consistently across multiple pages
        2. Detecting page numbers and running titles
        3. Removing detected elements while preserving document structure
        """
        if not text:
            return text
        
        lines = text.split('\n')
        if len(lines) < 2:
            # Not enough content to analyze
            return self._remove_basic_headers_footers(text)
        
        # Analyze line patterns across the document
        line_occurrence = {}
        top_line_occurrence = {}  # First few lines (potential headers)
        bottom_line_occurrence = {}  # Last few lines (potential footers)
        
        # Process page by page (separated by double newlines or detected page breaks)
        pages = self._split_into_pages(lines)
        
        for page_num, page_lines in enumerate(pages):
            # Analyze top lines (potential headers)
            for i, line in enumerate(page_lines[:3]):
                stripped = line.strip()
                if stripped and len(stripped) < 100:  # Headers are typically short
                    if stripped not in top_line_occurrence:
                        top_line_occurrence[stripped] = {'count': 0, 'positions': []}
                    top_line_occurrence[stripped]['count'] += 1
                    top_line_occurrence[stripped]['positions'].append((page_num, i))
            
            # Analyze bottom lines (potential footers)
            for i, line in enumerate(page_lines[-3:]):
                stripped = line.strip()
                if stripped and len(stripped) < 100:  # Footers are typically short
                    if stripped not in bottom_line_occurrence:
                        bottom_line_occurrence[stripped] = {'count': 0, 'positions': []}
                    bottom_line_occurrence[stripped]['count'] += 1
                    bottom_line_occurrence[stripped]['positions'].append((page_num, len(page_lines) - 3 + i))
            
            # Track all lines for consistency analysis
            for i, line in enumerate(page_lines):
                stripped = line.strip()
                if stripped and len(stripped) < 150:
                    if stripped not in line_occurrence:
                        line_occurrence[stripped] = {'count': 0, 'pages': set()}
                    line_occurrence[stripped]['count'] += 1
                    line_occurrence[stripped]['pages'].add(page_num)
        
        # Determine threshold for consistent content (appears on 70%+ of pages)
        num_pages = len(pages)
        header_threshold = max(2, int(num_pages * 0.7))  # At least 2 pages or 70%
        footer_threshold = max(2, int(num_pages * 0.7))
        
        # Identify headers to remove
        headers_to_remove = set()
        for line, data in top_line_occurrence.items():
            if data['count'] >= header_threshold:
                # Check if it's likely a header (not substantial content)
                if self._is_header_candidate(line, data['positions'], pages):
                    headers_to_remove.add(line)
        
        # Identify footers to remove
        footers_to_remove = set()
        for line, data in bottom_line_occurrence.items():
            if data['count'] >= footer_threshold:
                if self._is_footer_candidate(line, data['positions'], pages):
                    footers_to_remove.add(line)
        
        # Also detect and remove page numbers
        page_numbers_to_remove = set()
        for line, data in line_occurrence.items():
            if self._is_page_number(line, data['count'], num_pages):
                page_numbers_to_remove.add(line)
        
        # Combine all items to remove
        all_removals = headers_to_remove | footers_to_remove | page_numbers_to_remove
        
        # Remove detected headers/footers/page numbers
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped not in all_removals:
                cleaned_lines.append(line)
        
        # Post-processing: clean up empty lines
        cleaned_lines = self._clean_empty_lines(cleaned_lines)
        
        return '\n'.join(cleaned_lines)
    
    def _split_into_pages(self, lines: List[str]) -> List[List[str]]:
        """
        Split text into pages based on double newlines or other markers.
        """
        pages = []
        current_page = []
        
        for line in lines:
            # Detect page breaks (double newline or common page break patterns)
            if re.match(r'^Page\s+\d+', line) or re.match(r'^\s*[-—=]+\s*$', line):
                if current_page:
                    pages.append(current_page)
                current_page = []
            elif line.strip() == '' and current_page:
                # Single empty line - might be page break or paragraph break
                # Check if next line looks like a new page
                # For now, treat as potential paragraph break
                current_page.append(line)
            else:
                current_page.append(line)
        
        if current_page:
            pages.append(current_page)
        
        # If no clear page breaks detected, treat entire text as one page
        if not pages or len(pages) == 1:
            return [lines]
        
        return pages
    
    def _is_header_candidate(self, line: str, positions: List[Tuple[int, int]], 
                            pages: List[List[str]]) -> bool:
        """
        Determine if a line is likely a header based on content and position.
        """
        stripped = line.strip()
        
        # Check for common header patterns
        header_patterns = [
            r'^\d{4}\.\d{4,5}$',  # arXiv ID alone
            r'^arXiv:\d{4}\.\d{4,5}',  # arXiv ID with prefix
            r'^\[.*\]$',  # Square brackets (categories)
            r'^\d+\s*$',  # Page number alone
            r'^J\.?\s*\w+',  # Journal names
            r'^Proceedings?\s+of',  # Conference names
            r'^\w+\s+\d{4}$',  # Month Year
            r'^Submission\s+\d+',  # Submission info
            r'^\d+/\d+$',  # Volume/number
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        
        # Check line characteristics
        if len(stripped) < 50:  # Short lines are more likely headers
            # Check if it's mostly uppercase or numbers (running title)
            upper_ratio = sum(1 for c in stripped if c.isupper()) / max(len(stripped), 1)
            if upper_ratio > 0.5 or stripped.replace(' ', '').isdigit():
                return True
        
        return False
    
    def _is_footer_candidate(self, line: str, positions: List[Tuple[int, int]], 
                            pages: List[List[str]]) -> bool:
        """
        Determine if a line is likely a footer based on content and position.
        """
        stripped = line.strip()
        
        # Check for common footer patterns
        footer_patterns = [
            r'^Page\s+\d+$',  # Explicit page numbers
            r'^\d+\s*$',  # Page number alone
            r'^arXiv:\d{4}\.\d{4,5}',  # arXiv ID
            r'^Copyright\s+©',  # Copyright notice
            r'^\d{4}\s*[-–]\s*\d{4}$',  # Year range
            r'^doi:\s*\d+\.\d+',  # DOI prefix
            r'^\? ?\d+$',  # Question mark with number (anonymous review)
            r'^[A-Z][a-z]+\s+\d{4}$',  # Conference and year
        ]
        
        for pattern in footer_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        
        # Check if it's a page number pattern
        if stripped.isdigit() and len(stripped) <= 4:
            return True
        
        # Check line characteristics - footers are often short
        if len(stripped) < 30:
            # Check for copyright or footer-specific content
            footer_keywords = ['copyright', 'doi', 'accepted', 'published', 'submitted']
            if any(kw in stripped.lower() for kw in footer_keywords):
                return True
        
        return False
    
    def _is_page_number(self, line: str, count: int, total_pages: int) -> bool:
        """
        Determine if a line is a page number.
        """
        stripped = line.strip()
        
        # Explicit page number patterns
        explicit_patterns = [
            r'^Page\s+\d+$',
            r'^\[\s*\d+\s*\]$',
            r'^\d+\s*/\s*\d+$',  # Page X of Y format
        ]
        
        for pattern in explicit_patterns:
            if re.match(pattern, stripped):
                return True
        
        # Check for simple page numbers
        if stripped.isdigit():
            # Appears on most pages (page numbers)
            if count >= total_pages * 0.8:
                return True
            # Or it's a reasonable page number
            page_num = int(stripped)
            if 1 <= page_num <= total_pages * 2:  # Allow some buffer
                return True
        
        # Check for patterns like "1 / 10" or "Page 1"
        if re.match(r'^\d+\s*[/|-]?\s*\d+$', stripped):
            return True
        
        return False
    
    def _clean_empty_lines(self, lines: List[str]) -> List[str]:
        """
        Clean up consecutive empty lines while preserving paragraph breaks.
        """
        cleaned = []
        prev_empty = False
        
        for line in lines:
            if line.strip() == '':
                if not prev_empty:
                    cleaned.append('')
                    prev_empty = True
            else:
                cleaned.append(line)
                prev_empty = False
        
        return cleaned
    
    def _remove_basic_headers_footers(self, text: str) -> str:
        """
        Basic header/footer removal as fallback.
        """
        # Remove common arXiv headers/footers
        # Pattern: arXiv:XXXX.XXXXXvX [category] Date, Author
        text = re.sub(r'arXiv:\d+\.\d+v\d+.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers at bottom
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove common footer patterns
        text = re.sub(r'\n\s*arXiv\s+\d+\.\d+\s+.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*doi:\s*10\.\d+.*$', '', text, flags=re.IGNORECASE)
        
        # Clean up multiple empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _remove_headers_footers_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Remove headers and footers from extracted pages using positional information.
        
        This method uses the actual position data from PDF extraction to identify
        and remove headers and footers based on their location on the page.
        """
        if not pages:
            return pages
        
        cleaned_pages = []
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text:
                cleaned_pages.append(page)
                continue
            
            lines = page_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip lines that are likely headers/footers
                if not self._is_likely_header_footer(line, page.get('page', 0)):
                    cleaned_lines.append(line)
            
            page['text'] = '\n'.join(cleaned_lines)
            cleaned_pages.append(page)
        
        return cleaned_pages
    
    def _is_likely_header_footer(self, line: str, page_num: int) -> bool:
        """
        Determine if a line is likely a header or footer based on content.
        """
        stripped = line.strip()
        
        if not stripped or len(stripped) > 100:
            return False
        
        # Check for header/footer patterns
        patterns = [
            r'^arXiv:\d{4}\.\d+',
            r'^\[cs\.\w+\]',  # arXiv category
            r'^Page \d+',
            r'^\d+$',  # Simple page number
            r'^Proceedings? of',
            r'^International Conference',
            r'^Journal of',
            r'^Copyright',
        ]
        
        for pattern in patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        
        return False


def load_metadata(metadata_path: str) -> Dict[str, str]:
    """
    Load metadata from .txt file.
    
    Args:
        metadata_path: Path to metadata .txt file
        
    Returns:
        Dictionary with metadata fields
    """
    metadata = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    metadata[key] = value
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    
    return metadata

