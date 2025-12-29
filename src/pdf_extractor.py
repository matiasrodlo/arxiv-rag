"""
PDF Text Extraction Module with Multi-Library Fallback
Handles robust extraction from various PDF formats including scanned documents.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

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
    
    def __init__(self, enable_ocr: bool = True, ocr_language: str = "eng"):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
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
    
    def extract(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text and metadata from PDF.
        
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
        
        # Try each extraction method in order
        best_result = None
        best_score = 0
        
        for method in self.extraction_methods:
            try:
                extracted = method(str(pdf_path))
                if extracted and self._validate_extraction(extracted):
                    # Score the extraction quality
                    score = self._score_extraction_quality(extracted)
                    
                    # Keep the best extraction
                    if score > best_score:
                        best_score = score
                        best_result = extracted
                        
                    # If we get a very good extraction, use it immediately
                    if score >= 0.9:
                        result.update(extracted)
                        result['success'] = True
                        logger.info(f"Successfully extracted {pdf_path.name} using {result['method_used']} (quality: {score:.2f})")
                        return result
            except Exception as e:
                logger.warning(f"Extraction method {method.__name__} failed: {e}")
                continue
        
        # Use the best extraction found
        if best_result:
            result.update(best_result)
            result['success'] = True
            logger.info(f"Successfully extracted {pdf_path.name} using {result['method_used']} (quality: {best_score:.2f})")
            return result
        
        # Try OCR as last resort
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
        
        result['error'] = "All extraction methods failed"
        logger.error(f"Failed to extract text from {pdf_path.name}")
        return result
    
    def _extract_pymupdf(self, pdf_path: str) -> Optional[Dict]:
        """Extract using PyMuPDF (fitz) - best quality extraction."""
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
        
        # Use improved text extraction with layout preservation
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try multiple extraction strategies for better quality
            # Strategy 1: Standard extraction
            page_text = page.get_text()
            
            # Strategy 2: Layout-preserving extraction (better for multi-column)
            # Always try layout mode for better quality, not just for short text
            try:
                page_text_layout = page.get_text("layout")
                # Use layout if it's significantly better or if standard is short
                if len(page_text_layout) > len(page_text) * 1.1 or len(page_text) < 500:
                    page_text = page_text_layout
            except:
                pass
            
            # Strategy 3: Text blocks with coordinates (for better ordering)
            # Use blocks for better multi-column handling
            if len(page_text) < 1000:  # Try blocks if text seems incomplete
                try:
                    blocks = page.get_text("blocks")
                    if blocks:
                        # Sort blocks by position (top to bottom, left to right)
                        # Group blocks by vertical position (same line)
                        blocks_by_y = {}
                        for block in blocks:
                            y_pos = int(block[1] / 10)  # Group by 10-pixel bands
                            if y_pos not in blocks_by_y:
                                blocks_by_y[y_pos] = []
                            blocks_by_y[y_pos].append(block)
                        
                        # Sort by Y position, then X position within each line
                        sorted_blocks = []
                        for y_pos in sorted(blocks_by_y.keys()):
                            line_blocks = sorted(blocks_by_y[y_pos], key=lambda b: b[0])
                            sorted_blocks.extend(line_blocks)
                        
                        page_text_blocks = '\n'.join([b[4] for b in sorted_blocks if b[4].strip()])
                        if len(page_text_blocks) > len(page_text) * 0.9:  # Use if comparable or better
                            page_text = page_text_blocks
                except:
                    pass
            
            text_pages.append({
                'page': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text.append(page_text)
        
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
                # Try standard extraction
                page_text = page.extract_text() or ''
                
                # Always try layout mode for better quality
                try:
                    # Try extracting with layout preservation
                    page_text_layout = page.extract_text(layout=True) or ''
                    # Use layout if it's better or standard is poor
                    if len(page_text_layout) > len(page_text) * 1.05 or len(page_text) < 500:
                        page_text = page_text_layout
                except:
                    pass
                
                # Try extracting words with positions for better ordering
                if len(page_text) < 1000:
                    try:
                        words = page.extract_words()
                        if words:
                            # Sort words by position
                            words_sorted = sorted(words, key=lambda w: (w['top'], w['left']))
                            page_text_words = ' '.join([w['text'] for w in words_sorted])
                            if len(page_text_words) > len(page_text) * 0.9:
                                page_text = page_text_words
                    except:
                        pass
                
                # Extract tables if present (add as structured text)
                tables = page.extract_tables()
                if tables:
                    table_texts = []
                    for table in tables:
                        # Convert table to readable text
                        for row in table:
                            if row:
                                row_text = ' | '.join([str(cell) if cell else '' for cell in row])
                                table_texts.append(row_text)
                    if table_texts:
                        page_text += '\n\nTables:\n' + '\n'.join(table_texts)
                
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
            page_text = page.extract_text() or ''
            
            # If extraction is poor, try with layout hints
            if len(page_text) < 500 and hasattr(page, 'extract_text'):
                try:
                    # Try extraction with different parameters
                    page_text_alt = page.extract_text(extraction_mode="layout") or ''
                    if len(page_text_alt) > len(page_text):
                        page_text = page_text_alt
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
        """Extract using OCR (for scanned/image-based PDFs)."""
        images = convert_from_path(pdf_path)
        text_pages = []
        full_text = []
        
        for page_num, image in enumerate(images):
            page_text = pytesseract.image_to_string(image, lang=self.ocr_language)
            text_pages.append({
                'page': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text.append(page_text)
        
        return {
            'text': '\n\n'.join(full_text),
            'metadata': {'page_count': len(images), 'ocr_used': True},
            'pages': text_pages,
            'method_used': 'ocr'
        }
    
    def _validate_extraction(self, extracted: Dict, min_length: int = 100) -> bool:
        """Validate that extraction produced meaningful text."""
        text = extracted.get('text', '')
        if not text or len(text.strip()) < min_length:
            return False
        # Check if text is mostly whitespace or special characters
        if len(re.sub(r'\s+', '', text)) < min_length * 0.5:
            return False
        
        # Additional quality checks
        # Check for reasonable word count (not just symbols)
        words = text.split()
        if len(words) < min_length // 10:  # At least some words
            return False
        
        # Check for reasonable sentence structure
        sentences = text.split('.')
        if len(sentences) < 3:  # Should have at least a few sentences
            return False
        
        # Check for reasonable word-to-character ratio (should be mostly words, not symbols)
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 2:  # Too many single characters
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
        
        return text
    
    def _score_extraction_quality(self, extracted: Dict) -> float:
        """
        Score extraction quality (0-1).
        Higher score = better quality.
        """
        text = extracted.get('text', '')
        pages = extracted.get('pages', [])
        
        if not text or not pages:
            return 0.0
        
        score = 1.0
        
        # Penalize for very short text (more nuanced)
        if len(text) < 500:
            score -= 0.5
        elif len(text) < 1000:
            score -= 0.3
        elif len(text) < 3000:
            score -= 0.15
        elif len(text) < 5000:
            score -= 0.05
        
        # Penalize for empty pages
        empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 50)
        if empty_pages > 0:
            score -= (empty_pages / len(pages)) * 0.2
        
        # Penalize for low average chars per page
        if pages:
            avg_chars = sum(p.get('char_count', 0) for p in pages) / len(pages)
            if avg_chars < 500:
                score -= 0.2
            elif avg_chars < 1000:
                score -= 0.1
        
        # Check for common academic paper structure (more thorough)
        text_lower = text.lower()
        # Check abstract in first 3000 chars (more lenient)
        has_abstract = 'abstract' in text_lower[:3000] or 'summary' in text_lower[:3000]
        # Check introduction (more lenient)
        has_introduction = 'introduction' in text_lower or 'intro' in text_lower[:5000]
        # Check references (more lenient)
        has_references = 'reference' in text_lower or 'bibliography' in text_lower or 'references' in text_lower
        
        # Check for other academic indicators
        has_methods = 'method' in text_lower or 'methodology' in text_lower
        has_results = 'result' in text_lower or 'experiment' in text_lower
        has_conclusion = 'conclusion' in text_lower
        
        structure_elements = sum([has_abstract, has_introduction, has_references, has_methods, has_results, has_conclusion])
        structure_score = structure_elements / 6.0
        
        # Weight structure more if text is short (structure is more important indicator)
        if len(text) < 5000:
            score = score * 0.6 + structure_score * 0.4
        else:
            score = score * 0.7 + structure_score * 0.3
        
        return max(0.0, min(1.0, score))
    
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

