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

from ..extractors.formula_processor import FormulaProcessor, improve_formula_formatting


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
    
    def extract_sections(self, text: str, pages: Optional[List[Dict]] = None) -> List[Dict[str, any]]:
        """
        Extract sections from text based on common academic paper structure.
        
        Args:
            text: Full text of paper
            pages: Optional list of page dictionaries for page number mapping
            
        Returns:
            List of sections with title, content, and position information
        """
        import time
        import json
        from pathlib import Path
        
        DEBUG_LOG_PATH = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
        
        def debug_log(location, message, data, hypothesis_id=None):
            try:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000)
                }
                with open(DEBUG_LOG_PATH, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except:
                pass
        
        # #region agent log
        debug_log("text_processor.py:121", "extract_sections entry", {"text_length": len(text), "has_pages": pages is not None, "num_pages": len(pages) if pages else 0}, "A")
        # #endregion
        
        sections = []
        
        # Enhanced section patterns with more variations
        # Common section names with variations
        section_keywords = [
            'Abstract', 'Introduction', 'Background', 'Related Work', 'Related Works',
            'Methodology', 'Methods', 'Method', 'Approach', 'Approaches',
            'Implementation', 'Results', 'Result', 'Discussion', 'Discussions',
            'Conclusion', 'Conclusions', 'Summary', 'References', 'Bibliography',
            'Related Literature', 'Literature Review', 'Preliminaries', 'Preliminary',
            'Experimental Setup', 'Experiments', 'Evaluation', 'Analysis',
            'Future Work', 'Limitations', 'Acknowledgments', 'Acknowledgements',
            'Appendix', 'Appendices', 'Related', 'Work', 'Contributions'
        ]
        
        # Build pattern with all variations
        section_patterns = [
            # Exact matches (case-insensitive)
            r'^\s*(' + '|'.join(section_keywords) + r')\s*$',
            # With colon: "Introduction:"
            r'^\s*(' + '|'.join(section_keywords) + r')\s*:?\s*$',
            # Numbered sections: "1. Introduction", "1 Introduction"
            r'^\s*\d+[\.\)]\s*(' + '|'.join(section_keywords) + r')\s*$',
            r'^\s*\d+\.?\s+(' + '|'.join(section_keywords) + r')\s*$',
            # Roman numerals: "I. Introduction", "II. Methods"
            r'^\s*[IVX]+[\.\)]\s*(' + '|'.join(section_keywords) + r')\s*$',
            # ALL CAPS sections
            r'^\s*[A-Z][A-Z\s]{2,50}\s*$',
            # Numbered sections with any title: "1. Any Title Here"
            r'^\s*\d+[\.\)]\s+[A-Z][^\n]{2,80}$',
            # Section with number prefix: "Section 1: Introduction"
            r'^\s*Section\s+\d+\s*:?\s*[A-Z][^\n]{2,80}$',
        ]
        
        split_start = time.time()
        lines = text.split('\n')
        split_time = time.time() - split_start
        # #region agent log
        debug_log("text_processor.py:155", "Text split completed", {"time": split_time, "num_lines": len(lines)}, "D")
        # #endregion
        
        current_section = {'name': 'Introduction', 'text': '', 'start_char': 0, 'content_lines': []}
        char_position = 0
        
        # Pre-compile patterns for better performance
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in section_patterns]
        
        # Limit processing to first 5000 lines to avoid performance issues (reduced from 10000)
        max_lines = min(5000, len(lines))
        # #region agent log
        debug_log("text_processor.py:206", "Starting section loop", {"max_lines": max_lines, "total_lines": len(lines)}, "E")
        # #endregion
        
        last_log_time = time.time()
        for line_idx, line in enumerate(lines[:max_lines]):
            # Log progress every 1000 lines to detect if stuck
            if line_idx > 0 and line_idx % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                # #region agent log
                debug_log("text_processor.py:213", "Section loop progress", {"line_idx": line_idx, "elapsed_since_last": elapsed, "sections_found": len(sections)}, "E")
                # #endregion
                last_log_time = current_time
                
                # Safety: if processing more than 1 second per 1000 lines, something is wrong
                if elapsed > 1.0:
                    # #region agent log
                    debug_log("text_processor.py:213", "WARNING: Slow section processing", {"line_idx": line_idx, "elapsed": elapsed}, "E")
                    # #endregion
            line_stripped = line.strip()
            line_start_char = char_position
            
            # Check if line matches a section header (only check substantial lines)
            is_section_header = False
            # Skip lines that look like author names, emails, or other metadata
            # Section headers are usually short, don't contain emails, and match our patterns
            is_likely_section = (
                len(line_stripped) > 2 and len(line_stripped) < 100 and
                '@' not in line_stripped and  # Skip email addresses
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$', line_stripped) and  # Skip simple name patterns
                not re.match(r'^\d+$', line_stripped) and  # Skip pure numbers
                not re.match(r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+', line_stripped)  # Skip "First M. Last" patterns
            )
            
            if is_likely_section:
                # First try exact pattern matching
                for pattern in compiled_patterns:
                    match = pattern.match(line_stripped)
                    if match:
                        # Extract section name (handle groups)
                        section_name = line_stripped
                        if match.groups():
                            # Use the matched group if available
                            section_name = match.group(1) if match.group(1) else line_stripped
                        
                        # Normalize section name
                        section_name = section_name.strip().rstrip(':').strip()
                        
                        # Save previous section (page number will be calculated later in batch)
                        if current_section['content_lines']:
                            section_text = '\n'.join(current_section['content_lines']).strip()
                            if section_text:  # Only add non-empty sections
                                sections.append({
                                    'name': current_section['name'],
                                    'text': section_text,
                                    'start_char': current_section['start_char'],
                                    'end_char': char_position,
                                    'page': 1  # Temporary, will be updated in batch
                                })
                        
                        # Start new section
                        current_section = {
                            'name': section_name,
                            'text': '',
                            'start_char': char_position,
                            'content_lines': []
                        }
                        is_section_header = True
                        break
                
                # If no exact match, try fuzzy matching for common section names
                if not is_section_header:
                    line_lower = line_stripped.lower().rstrip(':').strip()
                    # Fuzzy match for common sections
                    fuzzy_sections = {
                        'abstract': 'Abstract',
                        'introduction': 'Introduction',
                        'background': 'Background',
                        'related work': 'Related Work',
                        'related works': 'Related Work',
                        'methodology': 'Methodology',
                        'methods': 'Methods',
                        'method': 'Methods',
                        'approach': 'Approach',
                        'implementation': 'Implementation',
                        'results': 'Results',
                        'result': 'Results',
                        'discussion': 'Discussion',
                        'conclusion': 'Conclusion',
                        'conclusions': 'Conclusion',
                        'references': 'References',
                        'bibliography': 'References',
                        'reference': 'References',
                        'works cited': 'References',
                        'appendix': 'Appendix',
                        'appendices': 'Appendix',
                        'supplementary': 'Appendix',
                        'supplement': 'Appendix',
                        'related literature': 'Related Work',
                        'literature review': 'Related Work',
                        'preliminaries': 'Preliminaries',
                        'experimental setup': 'Experiments',
                        'experiments': 'Experiments',
                        'evaluation': 'Evaluation',
                        'analysis': 'Analysis',
                        'future work': 'Future Work',
                        'limitations': 'Limitations',
                        'acknowledgments': 'Acknowledgments',
                        'acknowledgements': 'Acknowledgments',
                        'appendix': 'Appendix',
                        'appendices': 'Appendix'
                    }
                    
                    # Check for fuzzy match (exact or contains)
                    # Also check if line looks like references (starts with [1], [2], etc.)
                    if re.match(r'^\s*\[\d+\]', line_stripped):
                        # This looks like a reference - check if we're in references section
                        if 'reference' in current_section['name'].lower() or 'bibliography' in current_section['name'].lower():
                            # Already in references section
                            pass
                        else:
                            # Start new References section
                            if current_section['content_lines']:
                                section_text = '\n'.join(current_section['content_lines']).strip()
                                if section_text:
                                    sections.append({
                                        'name': current_section['name'],
                                        'text': section_text,
                                        'start_char': current_section['start_char'],
                                        'end_char': char_position,
                                        'page': 1  # Temporary, will be updated in batch
                                    })
                            current_section = {
                                'name': 'References',
                                'text': '',
                                'start_char': char_position,
                                'content_lines': []
                            }
                            is_section_header = True
                    
                    if not is_section_header:
                        for key, normalized_name in fuzzy_sections.items():
                            if line_lower == key or line_lower.startswith(key + ' ') or line_lower.endswith(' ' + key):
                                # Save previous section (page number will be calculated later in batch)
                                if current_section['content_lines']:
                                    section_text = '\n'.join(current_section['content_lines']).strip()
                                    if section_text:
                                        sections.append({
                                            'name': current_section['name'],
                                            'text': section_text,
                                            'start_char': current_section['start_char'],
                                            'end_char': char_position,
                                            'page': 1  # Temporary, will be updated in batch
                                        })
                                
                                # Start new section with normalized name
                                current_section = {
                                    'name': normalized_name,
                                    'text': '',
                                    'start_char': char_position,
                                    'content_lines': []
                                }
                                is_section_header = True
                                break
            
            if not is_section_header:
                current_section['content_lines'].append(line)
            
            # Update character position (include newline)
            char_position += len(line) + 1
            
            # Log progress every 1000 lines
            if line_idx > 0 and line_idx % 1000 == 0:
                # #region agent log
                debug_log("text_processor.py:195", "Section loop progress", {"lines_processed": line_idx, "sections_found": len(sections)}, "A")
                # #endregion
        
        # #region agent log
        debug_log("text_processor.py:200", "Section loop completed", {"total_lines": line_idx + 1, "sections_found": len(sections)}, "A")
        # #endregion
        
        # Add final section
        if current_section['content_lines']:
            section_text = '\n'.join(current_section['content_lines']).strip()
            sections.append({
                'name': current_section['name'],
                'text': section_text,
                'start_char': current_section['start_char'],
                'end_char': char_position,
                'page': 1  # Temporary, will be updated in batch
            })
        
        # Batch update page numbers for all sections (much faster than per-section lookups)
        if pages and sections:
            batch_start = time.time()
            # Build cache once
            if not hasattr(self, '_page_boundaries_cache') or self._page_boundaries_cache is None:
                page_boundaries = []
                current_pos = 0
                for page in pages:
                    page_text = page.get('text', '')
                    page_length = len(page_text)
                    page_boundaries.append((current_pos, current_pos + page_length, page.get('page', 1)))
                    current_pos += page_length + 2
                self._page_boundaries_cache = page_boundaries
            else:
                page_boundaries = self._page_boundaries_cache
            
            # Update all sections in one pass
            for section in sections:
                char_pos = section['start_char']
                # Binary search
                left, right = 0, len(page_boundaries) - 1
                while left <= right:
                    mid = (left + right) // 2
                    start, end, page_num = page_boundaries[mid]
                    if start <= char_pos < end:
                        section['page'] = page_num
                        break
                    elif char_pos < start:
                        right = mid - 1
                    else:
                        left = mid + 1
                else:
                    # Default to last page
                    section['page'] = page_boundaries[-1][2] if page_boundaries else 1
            
            batch_time = time.time() - batch_start
            # #region agent log
            debug_log("text_processor.py:400", "Batch page number update", {"time": batch_time, "num_sections": len(sections)}, "B")
            # #endregion
        
        # #region agent log
        debug_log("text_processor.py:222", "extract_sections exit", {"num_sections": len(sections)}, "A")
        # #endregion
        
        return sections
    
    def _find_page_for_position(self, char_position: int, pages: List[Dict]) -> int:
        """Find which page a character position belongs to (optimized with binary search)."""
        if not pages:
            return 1
        
        # Build page boundaries once (cached if called multiple times)
        if not hasattr(self, '_page_boundaries_cache') or self._page_boundaries_cache is None:
            page_boundaries = []
            current_pos = 0
            for page in pages:
                page_text = page.get('text', '')
                page_length = len(page_text)
                page_boundaries.append((current_pos, current_pos + page_length, page.get('page', 1)))
                current_pos += page_length + 2  # +2 for page separator
            self._page_boundaries_cache = page_boundaries
        else:
            page_boundaries = self._page_boundaries_cache
        
        # Binary search for page (fast - O(log n))
        left, right = 0, len(page_boundaries) - 1
        while left <= right:
            mid = (left + right) // 2
            start, end, page_num = page_boundaries[mid]
            if start <= char_position < end:
                return page_num
            elif char_position < start:
                right = mid - 1
            else:
                left = mid + 1
        
        # Default to last page if position is beyond
        return page_boundaries[-1][2] if page_boundaries else 1


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
        elif method == "semantic" and not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Auto-fallback to sentence chunking when semantic is requested but not available
            logger.warning("Semantic chunking requested but sentence-transformers not available. Falling back to sentence chunking.")
            self.method = "sentence"
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
    
    @staticmethod
    def extract_citations(text: str, sections: Optional[List[Dict]] = None) -> Dict:
        """
        Extract citations from text and parse reference section.
        Enhanced version with better context and page tracking.
        
        Args:
            text: Full text of the paper
            sections: List of section dictionaries with 'name' and 'text'
            
        Returns:
            Dictionary with 'in_text' and 'references' lists
        """
        if not text:
            return {'in_text': [], 'references': [], 'total_citations': 0, 'total_references': 0}
        
        citations_in_text = []
        references = []
        
        # Enhanced pattern to match citations: [1], [2, 3], [1-5], [1,2,3], etc.
        # Also handles (1), (2,3) style citations
        citation_patterns = [
            r'\[(\d+(?:[,\s-]\d+)*)\]',  # [1], [2, 3], [1-5]
            r'\((\d+(?:[,\s-]\d+)*)\)'   # (1), (2, 3) - less common but possible
        ]
        
        # Find all citations in text
        for pattern in citation_patterns:
            for match in re.finditer(pattern, text):
                citation_id = match.group(0)  # e.g., "[1]" or "[2, 3]"
                position = match.start()
                
                # Skip if this looks like part of a reference entry (e.g., "[1] Author...")
                # Check if it's in a references section
                is_in_references = False
                if sections:
                    for section in sections:
                        section_name = section.get('name', '').lower()
                        if any(term in section_name for term in ['reference', 'bibliography']):
                            start_char = section.get('start_char', 0)
                            end_char = section.get('end_char', len(text))
                            if start_char <= position < end_char:
                                is_in_references = True
                                break
                
                # Skip citations in references section (those are reference markers, not citations)
                if is_in_references:
                    continue
                
                # Get context around citation (100 chars before and after for better context)
                start_ctx = max(0, position - 100)
                end_ctx = min(len(text), position + len(citation_id) + 100)
                context = text[start_ctx:end_ctx]
                
                # Determine section
                section_name = 'Unknown'
                page_num = 1
                if sections:
                    for section in sections:
                        start_char = section.get('start_char', 0)
                        end_char = section.get('end_char', len(text))
                        if start_char <= position < end_char:
                            section_name = section.get('name', 'Unknown')
                            page_num = section.get('page', 1)
                            break
                
                # Extract individual citation numbers
                citation_numbers = []
                for num_str in re.findall(r'\d+', citation_id):
                    try:
                        num = int(num_str)
                        if 1 <= num <= 1000:  # Reasonable range for citations
                            citation_numbers.append(num)
                    except ValueError:
                        pass
                
                if citation_numbers:  # Only add if we found valid citation numbers
                    citations_in_text.append({
                        'citation_id': citation_id,
                        'position': position,
                        'context': context.strip(),
                        'section': section_name,
                        'page': page_num,
                        'citation_numbers': citation_numbers
                    })
        
        # Extract references section
        references_text = None
        if sections:
            for section in sections:
                section_name = section.get('name', '').lower()
                if any(term in section_name for term in ['reference', 'bibliography', 'works cited']):
                    references_text = section.get('text', '')
                    break
        
        # If no references section found, try to find it in text
        if not references_text:
            # Look for "References" or "Bibliography" heading
            ref_pattern = r'(?:References|Bibliography|Works Cited)[\s\n]+(.*?)(?=\n\s*\n|$)'
            match = re.search(ref_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                references_text = match.group(1)
        
        # Parse individual references if references section found
        if references_text:
            # Pattern to match reference entries: [1] Author, Title...
            ref_entry_pattern = r'\[(\d+)\]\s*(.+?)(?=\n\s*\[\d+\]|$)'
            for match in re.finditer(ref_entry_pattern, references_text, re.DOTALL):
                ref_id = match.group(1)
                ref_text = match.group(2).strip()
                
                # Try to extract title (usually in quotes or first line)
                title = None
                title_match = re.search(r'"([^"]+)"', ref_text)
                if not title_match:
                    title_match = re.search(r'([A-Z][^.]{10,100})', ref_text)
                if title_match:
                    title = title_match.group(1).strip()
                
                # Try to extract authors (usually before title, comma-separated)
                authors = []
                if title:
                    before_title = ref_text[:ref_text.find(title)]
                    # Look for author patterns
                    author_matches = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', before_title)
                    authors = author_matches[:5]  # Limit to first 5 authors
                
                # Try to extract year
                year = None
                year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
                if year_match:
                    year = int(year_match.group(0))
                
                # Try to extract arXiv ID if present
                arxiv_id = None
                arxiv_match = re.search(r'arXiv:\s*(\d{4}\.\d{4,5}v?\d*)', ref_text, re.IGNORECASE)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                
                references.append({
                    'ref_id': f"[{ref_id}]",
                    'text': ref_text[:500],  # Limit length
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'arxiv_id': arxiv_id
                })
        
        return {
            'in_text': citations_in_text,
            'references': references,
            'total_citations': len(citations_in_text),
            'total_references': len(references)
        }
    
    @staticmethod
    def extract_metadata(text: str, sections: Optional[List[Dict]] = None) -> Dict:
        """
        Extract title, authors, and abstract from text.
        
        Args:
            text: Full text of the paper
            sections: List of section dictionaries
            
        Returns:
            Dictionary with title, authors, abstract
        """
        metadata = {}
        
        # Extract title (usually first substantial line, before Abstract)
        lines = text.split('\n')
        title_candidates = []
        
        # First, try to find title in sections (some papers have title as a section)
        if sections:
            for section in sections:
                section_name = section.get('name', '').strip()
                section_text = section.get('text', '').strip()
                # Check if section name looks like a title
                if (len(section_name) > 10 and len(section_name) < 200 and 
                    section_name[0].isupper() and 
                    not any(word in section_name.lower() for word in ['abstract', 'introduction', 'method', 'result'])):
                    title_candidates.append((0, section_name))
                    break
        
        # If no title in sections, look in first lines
        if not title_candidates:
            for i, line in enumerate(lines[:30]):  # Check first 30 lines
                line = line.strip()
                if len(line) > 15 and len(line) < 250:  # Reasonable title length
                    # Skip common non-title patterns
                    if not re.match(r'^(arXiv:|Abstract|Introduction|1\.|I\.|Keywords|Index Terms)', line, re.IGNORECASE):
                        # Check if it looks like a title (capitalized, no ending punctuation, not all caps)
                        if (line[0].isupper() and 
                            not line.endswith('.') and 
                            not line.isupper() and  # Not all caps
                            line.count(':') <= 1):  # May have subtitle
                            title_candidates.append((i, line))
        
        if title_candidates:
            # Prefer first candidate (earliest in document)
            title = title_candidates[0][1]
            # Clean title (remove extra whitespace, fix common issues)
            title = re.sub(r'\s+', ' ', title).strip()
            metadata['title'] = title
        
        # Extract abstract
        abstract = None
        if sections:
            for section in sections:
                section_name = section.get('name', '').lower()
                if 'abstract' in section_name:
                    abstract = section.get('text', '').strip()
                    # Clean abstract (remove common prefixes)
                    abstract = re.sub(r'^abstract\s*:?\s*', '', abstract, flags=re.IGNORECASE)
                    # Remove arXiv ID if present at start
                    abstract = re.sub(r'^arXiv:\s*\d+\.\d+v?\d*\s*\[.*?\]\s*\d+\s+\w+\s+\d+\s*', '', abstract, flags=re.IGNORECASE)
                    break
        
        # If no abstract section found, try to find it in text
        if not abstract:
            # Try multiple patterns
            abstract_patterns = [
                r'Abstract\s*:?\s*(.+?)(?=\n\s*(?:1\.|Introduction|Keywords|Index Terms|I\.))',
                r'ABSTRACT\s*:?\s*(.+?)(?=\n\s*(?:1\.|Introduction|Keywords|Index Terms))',
                r'Abstract\s*:?\s*(.+?)(?=\n\n)',
            ]
            for pattern in abstract_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    # Clean abstract
                    abstract = re.sub(r'^abstract\s*:?\s*', '', abstract, flags=re.IGNORECASE)
                    abstract = re.sub(r'^arXiv:\s*\d+\.\d+v?\d*\s*\[.*?\]\s*\d+\s+\w+\s+\d+\s*', '', abstract, flags=re.IGNORECASE)
                    break
        
        if abstract:
            # Limit abstract length and clean
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            if len(abstract) > 2000:
                abstract = abstract[:2000] + '...'
            metadata['abstract'] = abstract
        
        # Extract authors (usually after title, before abstract)
        authors = []
        if metadata.get('title'):
            title_pos = text.find(metadata['title'])
            # Look for authors in next 20 lines after title
            after_title = text[title_pos + len(metadata['title']):title_pos + len(metadata['title']) + 2000]
            
            # Pattern: Author names (usually "First Last" or "First Middle Last")
            # Look for lines with multiple capitalized words
            author_lines = after_title.split('\n')[:15]
            for line in author_lines:
                line = line.strip()
                # Skip empty lines and common non-author patterns
                if not line or re.match(r'^(arXiv:|Abstract|1\.|Introduction|University|School|Department|Email)', line, re.IGNORECASE):
                    continue
                
                # Look for author patterns: "First Last" or "First M. Last"
                author_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)', line)
                if author_matches:
                    authors.extend(author_matches)
                    if len(authors) >= 10:  # Limit to 10 authors
                        break
        
        # Clean authors (remove duplicates, limit length)
        if authors:
            # Remove duplicates while preserving order
            seen = set()
            unique_authors = []
            for author in authors:
                author_clean = author.strip()
                if author_clean and author_clean not in seen:
                    seen.add(author_clean)
                    unique_authors.append(author_clean)
            metadata['authors'] = unique_authors[:10]  # Limit to 10 authors
        
        return metadata

