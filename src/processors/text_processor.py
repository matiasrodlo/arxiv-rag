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
        
        # Limit processing to first 10000 lines to avoid performance issues
        max_lines = min(10000, len(lines))
        # #region agent log
        debug_log("text_processor.py:165", "Starting section loop", {"max_lines": max_lines}, "A")
        # #endregion
        
        page_find_calls = 0
        for line_idx, line in enumerate(lines[:max_lines]):
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
                        
                        # Save previous section
                        if current_section['content_lines']:
                            section_text = '\n'.join(current_section['content_lines']).strip()
                            if section_text:  # Only add non-empty sections
                                page_find_start = time.time()
                                page_num = self._find_page_for_position(current_section['start_char'], pages) if pages else 1
                                page_find_time = time.time() - page_find_start
                                page_find_calls += 1
                                # #region agent log
                                debug_log("text_processor.py:175", "_find_page_for_position call", {"time": page_find_time, "call_number": page_find_calls, "section_name": current_section['name']}, "B")
                                # #endregion
                                
                                sections.append({
                                    'name': current_section['name'],
                                    'text': section_text,
                                    'start_char': current_section['start_char'],
                                    'end_char': char_position,
                                    'page': page_num
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
                                    page_num = self._find_page_for_position(current_section['start_char'], pages) if pages else 1
                                    sections.append({
                                        'name': current_section['name'],
                                        'text': section_text,
                                        'start_char': current_section['start_char'],
                                        'end_char': char_position,
                                        'page': page_num
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
                                # Save previous section
                                if current_section['content_lines']:
                                    section_text = '\n'.join(current_section['content_lines']).strip()
                                    if section_text:
                                        page_find_start = time.time()
                                        page_num = self._find_page_for_position(current_section['start_char'], pages) if pages else 1
                                        page_find_time = time.time() - page_find_start
                                        page_find_calls += 1
                                        # #region agent log
                                        debug_log("text_processor.py:310", "_find_page_for_position call (fuzzy)", {"time": page_find_time, "call_number": page_find_calls, "section_name": current_section['name']}, "B")
                                        # #endregion
                                        
                                        sections.append({
                                            'name': current_section['name'],
                                            'text': section_text,
                                            'start_char': current_section['start_char'],
                                            'end_char': char_position,
                                            'page': page_num
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
                debug_log("text_processor.py:195", "Section loop progress", {"lines_processed": line_idx, "sections_found": len(sections), "page_find_calls": page_find_calls}, "A")
                # #endregion
        
        # #region agent log
        debug_log("text_processor.py:200", "Section loop completed", {"total_lines": line_idx + 1, "sections_found": len(sections)}, "A")
        # #endregion
        
        # Add final section
        if current_section['content_lines']:
            join_start = time.time()
            section_text = '\n'.join(current_section['content_lines']).strip()
            join_time = time.time() - join_start
            # #region agent log
            debug_log("text_processor.py:207", "Final section join", {"time": join_time, "content_lines": len(current_section['content_lines'])}, "E")
            # #endregion
            
            page_find_start = time.time()
            page_num = self._find_page_for_position(current_section['start_char'], pages) if pages else 1
            page_find_time = time.time() - page_find_start
            page_find_calls += 1
            # #region agent log
            debug_log("text_processor.py:212", "_find_page_for_position call", {"time": page_find_time, "call_number": page_find_calls}, "B")
            # #endregion
            
            sections.append({
                'name': current_section['name'],
                'text': section_text,
                'start_char': current_section['start_char'],
                'end_char': char_position,
                'page': page_num
            })
        
        # #region agent log
        debug_log("text_processor.py:222", "extract_sections exit", {"num_sections": len(sections), "total_page_find_calls": page_find_calls}, "A")
        # #endregion
        
        return sections
    
    def _find_page_for_position(self, char_position: int, pages: List[Dict]) -> int:
        """Find which page a character position belongs to (optimized with binary search)."""
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
        
        if not pages:
            return 1
        
        # Build page boundaries once (cached if called multiple times)
        cache_build_start = time.time()
        if not hasattr(self, '_page_boundaries_cache') or self._page_boundaries_cache is None:
            # #region agent log
            debug_log("text_processor.py:220", "Building page boundaries cache", {"num_pages": len(pages)}, "B")
            # #endregion
            page_boundaries = []
            current_pos = 0
            for page in pages:
                page_text = page.get('text', '')
                page_length = len(page_text)
                page_boundaries.append((current_pos, current_pos + page_length, page.get('page', 1)))
                current_pos += page_length + 2  # +2 for page separator
            self._page_boundaries_cache = page_boundaries
            cache_build_time = time.time() - cache_build_start
            # #region agent log
            debug_log("text_processor.py:230", "Page boundaries cache built", {"time": cache_build_time, "num_boundaries": len(page_boundaries)}, "B")
            # #endregion
        else:
            page_boundaries = self._page_boundaries_cache
            # #region agent log
            debug_log("text_processor.py:234", "Using cached page boundaries", {"cache_size": len(page_boundaries)}, "B")
            # #endregion
        
        # Binary search for page
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

