"""
PDF Text Extraction Module with Multi-Library Fallback
Handles robust extraction from various PDF formats including scanned documents.
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


class PDFExtractor:
    """Extract text from PDFs with multiple fallback methods."""
    
    def __init__(self, enable_ocr: bool = True, ocr_language: str = "eng", max_retries: int = 2, 
                 enable_parallel: bool = True, max_workers: int = 4, large_pdf_threshold_mb: float = 20.0,
                 enable_caching: bool = True, cache_dir: Optional[str] = None):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        self.max_retries = max_retries
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.large_pdf_threshold_mb = large_pdf_threshold_mb
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.arxiv_rag_cache'
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
                # Remove large text from cache to save space, keep metadata
                cached_result = {
                    'metadata': result.get('metadata', {}),
                    'method_used': result.get('method_used'),
                    'quality_score': result.get('quality_score', 0.0),
                    'pdf_type': result.get('pdf_type', 'unknown'),
                    'num_pages': len(result.get('pages', [])),
                    'text_length': len(result.get('text', '')),
                    'cached': True
                }
                with open(cache_file, 'w') as f:
                    json.dump(cached_result, f)
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
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(pdf_path)
            cached = self._load_from_cache(cache_key)
            if cached:
                # For cached results, we still need to extract text (not cached to save space)
                # But we can skip if we only need metadata
                # For now, return cached metadata but note that full extraction needed
                logger.info(f"Found cached metadata for {pdf_path.name}, extracting text...")
                # Continue with extraction but use cached metadata if available
        
        # Check PDF size for optimization
        pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        is_large_pdf = pdf_size_mb > self.large_pdf_threshold_mb
        
        if is_large_pdf:
            logger.debug(f"Large PDF detected ({pdf_size_mb:.1f} MB), using optimized extraction")
        
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
                            
                        # Early exit optimization: if we get a very good extraction, use it immediately
                        # For large PDFs, be more lenient (0.85) to save time
                        # For very large PDFs (>50 MB or >50 pages), be even more lenient (0.80)
                        pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
                        page_count = extracted.get('pages', [])
                        num_pages = len(page_count) if page_count else 0
                        is_very_large = pdf_size_mb > 50.0 or num_pages > 50
                        
                        if is_very_large:
                            quality_threshold = 0.80  # Very lenient for very large PDFs
                        elif is_large_pdf:
                            quality_threshold = 0.85  # Lenient for large PDFs
                        else:
                            quality_threshold = 0.9   # Standard threshold
                        
                        if score >= quality_threshold:
                            result.update(extracted)
                            result['success'] = True
                            result['quality_score'] = score
                            logger.info(f"Successfully extracted {pdf_path.name} using {result['method_used']} (quality: {score:.2f})")
                            return result
                        break  # Success, no need to retry
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
        
        # Check if we should use parallel processing
        # Optimized thresholds: >30 pages OR >30 MB (reduces overhead for smaller PDFs)
        pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
        use_parallel = (self.enable_parallel and 
                       (len(doc) > 30 or pdf_size_mb > 30.0))  # Use parallel for PDFs with >30 pages OR >30MB
        
        if use_parallel:
            # Parallel page extraction for large PDFs
            text_pages = self._extract_pages_parallel(doc)
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
        if not page_text or len(page_text) < 500:
            try:
                page_text_layout = page.get_text("layout")
                if page_text_layout:
                    if not page_text or len(page_text_layout) > len(page_text) * 1.05:
                        page_text = page_text_layout
            except Exception as e:
                logger.debug(f"Layout extraction failed for page {page_num + 1}: {e}")
        
        # Strategy 3: Standard extraction as fallback
        if not page_text:
            try:
                page_text = page.get_text()
            except Exception as e:
                logger.debug(f"Standard extraction failed for page {page_num + 1}: {e}")
                page_text = ''
        
        # Strategy 4: Extract tables using PyMuPDF (if available)
        # Try to find and extract tables to enhance content
        try:
            tables = self._extract_tables_pymupdf(page)
            if tables:
                table_text = self._format_tables_markdown(tables)
                if table_text:
                    if page_text:
                        page_text += '\n\n' + table_text
                    else:
                        page_text = table_text
        except Exception as e:
            logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
        
        # Strategy 5: Text blocks with coordinates (for better multi-column handling)
        # Use blocks for better multi-column handling if text seems incomplete
        if len(page_text) < 1000:
            try:
                blocks = page.get_text("blocks")
                if blocks:
                    page_text_blocks = self._reconstruct_text_from_blocks(blocks)
                    if page_text_blocks and len(page_text_blocks) > len(page_text) * 0.9:
                        page_text = page_text_blocks
            except Exception as e:
                logger.debug(f"Block extraction failed for page {page_num + 1}: {e}")
        
        # Strategy 6: Try rawdict for very structured documents (skip for parallel to save time)
        if len(page_text) < 500 and not self.enable_parallel:
            try:
                rawdict = page.get_text("rawdict")
                if rawdict and rawdict.get('blocks'):
                    page_text_raw = self._reconstruct_text_from_dict(rawdict)
                    if page_text_raw and len(page_text_raw) > len(page_text):
                        page_text = page_text_raw
            except Exception as e:
                logger.debug(f"Rawdict extraction failed for page {page_num + 1}: {e}")
        
        return page_text
    
    def _extract_tables_pymupdf(self, page) -> List[List[List[str]]]:
        """Extract tables from a PyMuPDF page using find_tables."""
        tables = []
        try:
            # PyMuPDF's find_tables method (if available in version)
            if hasattr(page, 'find_tables'):
                table_list = page.find_tables()
                for table in table_list:
                    try:
                        # Extract table data
                        table_data = table.extract()
                        if table_data and len(table_data) > 0:
                            tables.append(table_data)
                    except Exception as e:
                        logger.debug(f"Error extracting table data: {e}")
                        continue
        except Exception as e:
            logger.debug(f"find_tables not available or failed: {e}")
            # Fallback: Try to detect tables from text blocks
            try:
                blocks = page.get_text("blocks")
                if blocks:
                    # Look for table-like structures (rectangular blocks with aligned text)
                    table_blocks = self._detect_table_blocks(blocks)
                    if table_blocks:
                        tables.extend(table_blocks)
            except Exception as e2:
                logger.debug(f"Table block detection failed: {e2}")
        
        return tables
    
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
        """Reconstruct text from PyMuPDF blocks with improved multi-column handling."""
        if not blocks:
            return ''
        
        # Improved block sorting for multi-column layouts
        # Group blocks by vertical position with adaptive tolerance
        blocks_by_y = {}
        for block in blocks:
            if len(block) < 5 or not block[4].strip():  # block[4] is the text
                continue
            
            # Use adaptive grouping based on font size (if available)
            # For now, use 15-pixel bands for better line grouping
            y_pos = int(block[1] / 15)  # Group by 15-pixel bands
            if y_pos not in blocks_by_y:
                blocks_by_y[y_pos] = []
            blocks_by_y[y_pos].append(block)
        
        # Sort blocks: top to bottom, then left to right within each line
        sorted_blocks = []
        for y_pos in sorted(blocks_by_y.keys()):
            line_blocks = sorted(blocks_by_y[y_pos], key=lambda b: b[0])  # Sort by x-coordinate
            sorted_blocks.extend(line_blocks)
        
        # Reconstruct text with proper spacing
        text_parts = []
        prev_block = None
        for block in sorted_blocks:
            block_text = block[4].strip()
            if not block_text:
                continue
            
            # Add spacing based on block positions
            if prev_block:
                # Calculate distance between blocks
                x_gap = block[0] - (prev_block[0] + prev_block[2])  # Current x - (prev x + prev width)
                y_gap = block[1] - (prev_block[1] + prev_block[3])  # Current y - (prev y + prev height)
                
                # If significant vertical gap, add newline
                if y_gap > 10:  # More than 10 pixels vertical gap
                    text_parts.append('\n')
                # If significant horizontal gap, add space (likely new column or word)
                elif x_gap > 20:  # More than 20 pixels horizontal gap
                    text_parts.append(' ')
            
            text_parts.append(block_text)
            prev_block = block
        
        return ' '.join(text_parts)
    
    def _extract_and_format_tables_pdfplumber(self, tables: List, page) -> str:
        """Extract and format tables from pdfplumber with enhanced formatting."""
        if not tables:
            return ''
        
        formatted_tables = []
        
        for table_idx, table in enumerate(tables, 1):
            if not table:
                continue
            
            # Try to get table bounding box for caption detection
            try:
                # Get table bounding box if available
                table_bbox = None
                if hasattr(page, 'find_tables'):
                    table_objects = page.find_tables()
                    if table_idx <= len(table_objects):
                        table_obj = table_objects[table_idx - 1]
                        if hasattr(table_obj, 'bbox'):
                            table_bbox = table_obj.bbox
            except:
                pass
            
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
            
            for page_num, page in enumerate(pdf.pages):
                page_text = ''
                
                # Strategy 1: Try layout mode first (better for multi-column)
                try:
                    page_text_layout = page.extract_text(layout=True) or ''
                    if page_text_layout:
                        page_text = page_text_layout
                except Exception as e:
                    logger.debug(f"Layout extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 2: Standard extraction as comparison
                if not page_text or len(page_text) < 500:
                    try:
                        page_text_standard = page.extract_text() or ''
                        if page_text_standard and len(page_text_standard) > len(page_text) * 1.05:
                            page_text = page_text_standard
                    except Exception as e:
                        logger.debug(f"Standard extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 3: Word-level extraction with improved ordering (for complex layouts)
                if len(page_text) < 1000:
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
                        logger.debug(f"Word extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 4: Extract tables with enhanced formatting
                try:
                    tables = page.extract_tables()
                    if tables:
                        # Enhanced table extraction with better formatting
                        table_text = self._extract_and_format_tables_pdfplumber(tables, page)
                        if table_text:
                            if page_text:
                                page_text += '\n\n' + table_text
                            else:
                                page_text = table_text
                except Exception as e:
                    logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
                
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
                'method_used': 'pdfplumber'
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
            # Use higher DPI for better quality, but adaptive based on page size
            images = convert_from_path(pdf_path, dpi=300)
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
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                # Sharpen image
                image = image.filter(ImageFilter.SHARPEN)
            except Exception as e:
                logger.debug(f"Image preprocessing failed for page {page_num + 1}: {e}")
            
            # Try multiple PSM modes and use the best result
            psm_modes = [
                ('6', 'Uniform block of text'),  # Default for most academic papers
                ('3', 'Fully automatic page segmentation'),  # Good for complex layouts
                ('4', 'Single column of text'),  # Good for single column
                ('11', 'Sparse text'),  # Good for sparse text
            ]
            
            best_text = ''
            best_length = 0
            
            for psm_mode, description in psm_modes:
                try:
                    # Build config string with whitelist
                    whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}\'"-+=*/<>%&|\\@#$^_~`'
                    config_str = f'--psm {psm_mode} -c tessedit_char_whitelist={whitelist}'
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
        """
        if not text:
            return text
        
        # Fix common extraction artifacts
        
        # 1. Fix broken sentences (period followed by lowercase)
        text = re.sub(r'\.\s+([a-z])', r'. \1', text)  # Ensure space after period
        
        # 2. Fix broken words across lines (hyphen at end of line)
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove line breaks in hyphenated words
        
        # 3. Fix broken URLs and emails
        text = re.sub(r'([a-zA-Z0-9])\s+([@.])\s+([a-zA-Z0-9])', r'\1\2\3', text)
        
        # 4. Fix broken mathematical expressions
        # Fix spacing around operators
        text = re.sub(r'([a-zA-Z0-9])\s*([+\-*/=<>])\s*([a-zA-Z0-9])', r'\1 \2 \3', text)
        # But preserve subscripts/superscripts
        text = re.sub(r'([a-zA-Z0-9])\s*_\s*([a-zA-Z0-9])', r'\1_\2', text)
        text = re.sub(r'([a-zA-Z0-9])\s*\^\s*([a-zA-Z0-9])', r'\1^\2', text)
        
        # 5. Fix broken citations [1] -> [1] (remove spaces)
        text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
        
        # 6. Fix broken references (e.g., "Figure 1" split across lines)
        text = re.sub(r'(Figure|Table|Equation|Section|Algorithm)\s+\n\s*(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
        
        # 7. Fix broken abbreviations (e.g., "e. g." -> "e.g.")
        text = re.sub(r'\b([a-z])\.\s+([a-z])\.', r'\1.\2.', text)
        
        # 8. Fix multiple spaces (but preserve intentional spacing)
        text = re.sub(r' {2,}', ' ', text)
        
        # 9. Fix broken parentheses and brackets
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\[\s+', '[', text)
        text = re.sub(r'\s+\]', ']', text)
        
        # 10. Fix broken quotes
        text = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', text)
        text = re.sub(r"'\s+([^']+)\s+'", r"'\1'", text)
        
        # 11. Normalize line breaks (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        
        # 12. Fix broken decimal numbers
        text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
        
        # 13. Fix broken percentages
        text = re.sub(r'(\d)\s+%', r'\1%', text)
        
        # 14. Fix broken units (e.g., "5 m s" -> "5 ms")
        text = re.sub(r'(\d)\s+([a-z]{1,3})\s+([a-z])', r'\1 \2\3', text)
        
        # 15. Fix broken LaTeX commands
        text = re.sub(r'\\\s+([a-z]+)', r'\\\1', text)  # Fix \ command spacing
        text = re.sub(r'\\\s*\{', r'\\{', text)  # Fix \ { -> \{
        
        # 16. Fix broken equation numbers
        text = re.sub(r'\((\d+)\)\s*$', r'(\1)', text, flags=re.MULTILINE)  # Fix (1) spacing
        
        # 17. Fix broken figure/table references
        text = re.sub(r'(Figure|Table|Fig\.|Tab\.)\s+(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
        
        # 18. Fix broken section references
        text = re.sub(r'Section\s+(\d+)', r'Section \1', text, flags=re.IGNORECASE)
        
        # 19. Fix broken equation references
        text = re.sub(r'Equation\s+\((\d+)\)', r'Equation (\1)', text, flags=re.IGNORECASE)
        
        # 20. Fix broken author names (common in headers)
        # Pattern: LastName FirstName (should be LastName, FirstName or LastName FirstName)
        # This is tricky, so we'll be conservative
        
        # 21. Fix broken dates
        text = re.sub(r'(\d{4})\s+([A-Z][a-z]+)\s+(\d{1,2})', r'\1 \2 \3', text)  # Year Month Day
        
        # 22. Fix broken version numbers
        text = re.sub(r'v\s*(\d+)', r'v\1', text)  # v 1 -> v1
        
        # 23. Remove excessive whitespace at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # 24. Fix broken compound words (common in technical text)
        # Pattern: word1 word2 where they should be together
        # This is very context-dependent, so we'll be conservative
        
        # 25. Fix broken page breaks in the middle of words
        # Remove single-character words at line ends (likely broken words)
        text = re.sub(r'\b([a-zA-Z])\s+\n\s+([a-zA-Z])', r'\1\2', text)
        
        # 26. Fix broken hyphenation at line breaks
        # Pattern: word-\nword -> word-word
        text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-zA-Z])', r'\1\2', text)
        
        # 27. Fix broken spaces in numbers (e.g., "1 000" -> "1000" for thousands)
        # But be careful - this might be intentional formatting
        # Only fix if it's clearly wrong (single space in middle of number)
        text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)  # Fix "1 000" -> "1000"
        
        # 28. Fix broken special characters
        text = re.sub(r'&amp;', '&', text)  # HTML entities
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # 29. Fix broken em/en dashes
        text = re.sub(r'--', '—', text)  # Double hyphen to em dash
        text = re.sub(r' - ', ' – ', text)  # Space-hyphen-space to en dash
        
        # 30. Normalize quotes (smart quotes to regular quotes)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # 31. Fix broken multi-column text (common issue)
        # Pattern: word at end of line, single character, then word (likely column break)
        # Be conservative - only fix obvious cases
        text = re.sub(r'([a-zA-Z]{3,})\s+([a-zA-Z])\s+([a-zA-Z]{3,})', 
                     lambda m: m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) 
                     if len(m.group(2)) == 1 and m.group(1)[-1].islower() and m.group(3)[0].islower()
                     else m.group(0), text)
        
        # 32. Fix broken references and citations (improved)
        # Pattern: [1] with spaces -> [1]
        text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
        # Pattern: (Author, Year) with broken spacing
        text = re.sub(r'\(\s*([A-Z][a-z]+)\s*,\s*(\d{4})\s*\)', r'(\1, \2)', text)
        
        # 33. Fix broken section numbers (e.g., "Section 3 . 2" -> "Section 3.2")
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        
        # 34. Fix broken equation references (e.g., "Equation ( 1 )" -> "Equation (1)")
        text = re.sub(r'\((\s*\d+\s*)\)', lambda m: '(' + m.group(1).strip() + ')', text)
        
        # 35. Fix broken figure/table captions
        # Pattern: "Figure 1 : Description" -> "Figure 1: Description"
        text = re.sub(r'(Figure|Table|Fig\.|Tab\.)\s+(\d+)\s*:\s*', r'\1 \2: ', text, flags=re.IGNORECASE)
        
        # 36. Fix broken list items (numbered and bulleted)
        text = re.sub(r'^\s*(\d+)\s*\.\s+([A-Z])', r'\1. \2', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-•]\s+([A-Z])', r'- \1', text, flags=re.MULTILINE)
        
        # 37. Fix broken footnotes (e.g., "text1" -> "text¹")
        # This is complex, so we'll be conservative
        
        # 38. Fix broken Greek letters and special symbols (common in academic papers)
        # Pattern: broken alpha, beta, etc. (these are often single characters, so hard to fix)
        
        # 39. Fix broken spacing in mathematical expressions
        # Pattern: "x = y" with broken spacing -> "x = y"
        text = re.sub(r'([a-zA-Z0-9])\s*=\s*([a-zA-Z0-9])', r'\1 = \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s*<\s*([a-zA-Z0-9])', r'\1 < \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s*>\s*([a-zA-Z0-9])', r'\1 > \2', text)
        
        # 40. Final cleanup: remove excessive whitespace while preserving structure
        # Remove spaces at start/end of lines
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines but preserve paragraph breaks
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')  # Preserve single empty line
                prev_empty = True
        
        text = '\n'.join(cleaned_lines)
        
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
    
    def clean_text(self, text: str, remove_headers_footers: bool = True) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            remove_headers_footers: Remove common headers/footers
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        if remove_headers_footers:
            # Remove common arXiv headers/footers
            # Pattern: arXiv:XXXX.XXXXXvX [category] Date, Author
            text = re.sub(r'arXiv:\d+\.\d+v\d+.*?\n', '', text, flags=re.IGNORECASE)
            # Remove page numbers at bottom
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Fix common encoding issues
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')
        
        return text.strip()


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

