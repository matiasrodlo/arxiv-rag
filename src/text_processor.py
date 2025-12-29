"""
Text Processing and Chunking Module
Handles text cleaning, normalization, and intelligent chunking.
"""

import re
from typing import List, Dict, Optional
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

from .formula_processor import FormulaProcessor, improve_formula_formatting


class TextProcessor:
    """Process and clean extracted text."""
    
    def __init__(self, 
                 remove_headers_footers: bool = True,
                 normalize_whitespace: bool = True,
                 fix_encoding: bool = True,
                 improve_formulas: bool = True):
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding
        self.improve_formulas = improve_formulas
        
        if self.improve_formulas:
            self.formula_processor = FormulaProcessor(preserve_latex=True, normalize_spacing=True)
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues
        if self.fix_encoding:
            text = text.replace('\x00', '')
            text = text.replace('\ufffd', '')
            # Remove other control characters except newlines and tabs
            text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # Remove headers/footers
        if self.remove_headers_footers:
            # Remove arXiv headers (multiple patterns to handle variations)
            # Pattern 1: arXiv:XXXX.XXXXXvX [category] Date (with brackets)
            text = re.sub(r'arXiv:\d+\.\d+v\d+\s*\[[^\]]+\]\s*\d+\s+\w+\s+\d+', '', text, flags=re.IGNORECASE)
            # Pattern 2: arXiv:XXXX.XXXXXvX followed by any text until newline
            text = re.sub(r'arXiv:\d+\.\d+v\d+[^\n]*', '', text, flags=re.IGNORECASE)
            # Pattern 3: Standalone arXiv line
            text = re.sub(r'^\s*arXiv:.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove paper title repetition in headers (appears on page headers)
            # Look for title-like patterns that repeat (usually first line of paper)
            lines = text.split('\n')
            if len(lines) > 5:
                # Check first few lines for potential title (first substantial line)
                first_lines = [l.strip() for l in lines[:10] if l.strip() and len(l.strip()) > 15]
                if first_lines:
                    potential_title = first_lines[0]
                    # Count how many times it appears
                    title_count = text.lower().count(potential_title.lower())
                    
                    # If title appears many times (more than 3), it's likely in page headers
                    if len(potential_title) > 15 and title_count > 3:
                        # Remove duplicate occurrences that appear on their own line
                        # Pattern: title on its own line, possibly followed by a number
                        # We'll remove all but the first occurrence
                        title_pattern = re.escape(potential_title)
                        # Find all occurrences
                        matches = list(re.finditer(rf'^{title_pattern}\s*\d*\s*$', text, flags=re.IGNORECASE | re.MULTILINE))
                        
                        # Keep the first occurrence (likely the actual title), remove the rest
                        if len(matches) > 1:
                            # Remove from end to start to preserve indices
                            for match in reversed(matches[1:]):
                                start, end = match.span()
                                # Remove the line including newlines
                                text = text[:start] + text[end:]
                                # Remove extra newlines
                                text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Remove standalone page numbers (more aggressive patterns)
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            # Remove page numbers with "Page" prefix
            text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE | re.MULTILINE)
            # Remove page numbers at end of lines (common in footers)
            text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
            
            # Remove common footer patterns
            text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r' +', ' ', text)
            # Replace multiple newlines with double newline (paragraph break)
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove trailing whitespace from lines
            text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Improve formula formatting
        if self.improve_formulas:
            text = improve_formula_formatting(text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> List[Dict[str, any]]:
        """
        Extract sections from text based on common academic paper structure.
        
        Args:
            text: Full text of paper
            
        Returns:
            List of sections with title and content
        """
        sections = []
        
        # Common section patterns
        section_patterns = [
            r'^\s*(Abstract|Introduction|Background|Related Work|Methodology|Methods|Method|Approach|Implementation|Results|Discussion|Conclusion|Conclusions|References|Bibliography)\s*$',
            r'^\s*\d+\.\s+[A-Z][^\n]+$',  # Numbered sections: "1. Introduction"
            r'^\s*[A-Z][A-Z\s]+\s*$',  # ALL CAPS sections
        ]
        
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': []}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line matches a section header
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section
                    if current_section['content']:
                        sections.append({
                            'title': current_section['title'],
                            'content': '\n'.join(current_section['content']).strip()
                        })
                    
                    # Start new section
                    current_section = {
                        'title': line_stripped,
                        'content': []
                    }
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content']).strip()
            })
        
        return sections


class TextChunker:
    """Intelligent text chunking with multiple strategies."""
    
    def __init__(self,
                 method: str = "semantic",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 model_name: Optional[str] = None,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000):
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Load model for semantic chunking
        if method == "semantic" and SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded model for semantic chunking: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}. Falling back to fixed chunking.")
                self.method = "fixed"
                self.model = None
        else:
            self.model = None
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunks with text and metadata
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        if self.method == "semantic" and self.model:
            return self._chunk_semantic(text, metadata)
        elif self.method == "sentence":
            return self._chunk_by_sentence(text, metadata)
        else:
            return self._chunk_fixed(text, metadata)
    
    def _chunk_fixed(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """Fixed-size chunking with overlap."""
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence ending within last 20% of chunk
                lookback_start = max(start, end - int(self.chunk_size * 0.2))
                sentence_end = max(
                    text.rfind('. ', lookback_start, end),
                    text.rfind('.\n', lookback_start, end),
                    text.rfind('.\n\n', lookback_start, end)
                )
                if sentence_end > lookback_start:
                    end = sentence_end + 1
                    chunk_text = text[start:end]
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'chunk_method': 'fixed',
                    'char_start': start,
                    'char_end': end
                })
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        return chunks
    
    def _chunk_by_sentence(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """Chunk by sentences, grouping to reach target size."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_metadata = (metadata or {}).copy()
                    chunk_metadata.update({
                        'chunk_index': len(chunks),
                        'chunk_method': 'sentence'
                    })
                    chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': chunk_metadata
                    })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'chunk_method': 'sentence'
                })
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def _chunk_semantic(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Semantic chunking using sentence embeddings.
        Groups sentences with similar embeddings together.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        if len(sentences) < 2:
            # Fall back to fixed chunking for very short texts
            return self._chunk_fixed(text, metadata)
        
        # Get embeddings for all sentences
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Falling back to fixed chunking.")
            return self._chunk_fixed(text, metadata)
        
        # Group sentences based on similarity
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence_length = len(sentence)
            
            # Check if we should start a new chunk
            should_break = False
            
            if current_length + sentence_length > self.max_chunk_size:
                should_break = True
            elif current_chunk and i > 0:
                # Check semantic similarity with previous sentence
                prev_embedding = embeddings[i - 1]
                similarity = self._cosine_similarity(embedding, prev_embedding)
                # Break if similarity drops significantly (semantic shift)
                if similarity < 0.7:  # Threshold for semantic boundary
                    should_break = True
            
            if should_break and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_metadata = (metadata or {}).copy()
                    chunk_metadata.update({
                        'chunk_index': len(chunks),
                        'chunk_method': 'semantic'
                    })
                    chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': chunk_metadata
                    })
                
                # Start new chunk with overlap
                overlap_size = min(len(current_chunk), max(1, int(len(current_chunk) * 0.2)))
                current_chunk = current_chunk[-overlap_size:]
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'chunk_method': 'semantic'
                })
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    @staticmethod
    def _cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

