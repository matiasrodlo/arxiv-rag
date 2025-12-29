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
            if len(page_text) < 500:  # If standard extraction is short, try layout
                try:
                    page_text_layout = page.get_text("layout")
                    if len(page_text_layout) > len(page_text):
                        page_text = page_text_layout
                except:
                    pass
            
            # Strategy 3: Text blocks with coordinates (for better ordering)
            if len(page_text) < 500:
                try:
                    blocks = page.get_text("blocks")
                    if blocks:
                        # Sort blocks by position (top to bottom, left to right)
                        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
                        page_text_blocks = '\n'.join([b[4] for b in blocks_sorted if b[4].strip()])
                        if len(page_text_blocks) > len(page_text):
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
        
        return {
            'text': '\n\n'.join(full_text),
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
                
                # If poor extraction, try with layout
                if len(page_text) < 500:
                    try:
                        # Try extracting with layout preservation
                        page_text_layout = page.extract_text(layout=True) or ''
                        if len(page_text_layout) > len(page_text):
                            page_text = page_text_layout
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
            
            return {
                'text': '\n\n'.join(full_text),
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
            # Fix broken words (common in pypdf)
            page_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', page_text)  # Fix camelCase breaks
            # Fix broken numbers
            page_text = re.sub(r'(\d)\s+(\d)', r'\1\2', page_text)  # Fix number breaks
            
            text_pages.append({
                'page': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text.append(page_text)
        
        return {
            'text': '\n\n'.join(full_text),
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
        
        return True
    
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
        
        # Penalize for very short text
        if len(text) < 1000:
            score -= 0.3
        elif len(text) < 5000:
            score -= 0.1
        
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
        
        # Check for common academic paper structure
        text_lower = text.lower()
        has_abstract = 'abstract' in text_lower[:2000]
        has_introduction = 'introduction' in text_lower
        has_references = 'reference' in text_lower or 'bibliography' in text_lower
        
        structure_score = sum([has_abstract, has_introduction, has_references]) / 3.0
        score = score * 0.7 + structure_score * 0.3  # Weight structure
        
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

