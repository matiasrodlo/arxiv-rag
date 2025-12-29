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
        for method in self.extraction_methods:
            try:
                extracted = method(str(pdf_path))
                if extracted and self._validate_extraction(extracted):
                    result.update(extracted)
                    result['success'] = True
                    logger.info(f"Successfully extracted {pdf_path.name} using {result['method_used']}")
                    return result
            except Exception as e:
                logger.warning(f"Extraction method {method.__name__} failed: {e}")
                continue
        
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
        """Extract using PyMuPDF (fitz)."""
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
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
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
                page_text = page.extract_text() or ''
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
        """Extract using pypdf (lightweight fallback)."""
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
            page_text = page.extract_text() or ''
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
        return True
    
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

