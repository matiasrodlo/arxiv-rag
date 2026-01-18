#!/usr/bin/env python3
"""
Comprehensive chunk quality improvement script.
Implements all advanced improvements:
1. Sentence-aware chunking
2. Section-aware chunking (respects IMRaD boundaries)
3. Citation preservation
4. Quality filtering
5. Enhanced metadata
6. Abstract preservation
7. Adaptive overlap
8. Semantic coherence (optional, requires sentence-transformers)
9. Named Entity Recognition (NER) - extracts entities from chunks
10. Paragraph-aware chunking - respects paragraph boundaries
11. Keyword extraction - identifies important terms
12. Table/Figure reference detection
13. Deduplication
14. Text normalization for embeddings - optimizes text for embedding models
15. Token-based chunk sizing - model-specific optimization for embedding models
16. Context window optimization - adds section headers and paper title to chunks
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import argparse
import multiprocessing as mp
from functools import partial

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using regex-based sentence detection (less accurate).")
    print("Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class AdvancedChunkImprover:
    def __init__(self, 
                 target_chunk_size: int = 950,
                 overlap_size: int = 100,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1500,
                 use_spacy: bool = True,
                 use_semantic: bool = False,
                 min_quality_score: float = 0.5,
                 preserve_abstract: bool = True,
                 adaptive_overlap: bool = True,
                 use_tokens: bool = False,
                 embedding_model: str = 'openai',
                 context_optimization: bool = True):
        """
        Initialize advanced chunk improver with all enhancements.
        
        Args:
            target_chunk_size: Target chunk size in characters or tokens (default: 950)
            overlap_size: Base overlap size in characters or tokens (default: 100)
            min_chunk_size: Minimum chunk size (default: 200)
            max_chunk_size: Maximum chunk size (default: 1500)
            use_spacy: Whether to use spaCy for sentence detection (default: True)
            use_semantic: Whether to use semantic similarity for grouping (default: False)
            min_quality_score: Minimum quality score to keep chunk (0-1, default: 0.5)
            preserve_abstract: Whether to preserve abstract as complete chunk (default: True)
            adaptive_overlap: Whether to use adaptive overlap at section boundaries (default: True)
            use_tokens: Whether to use token-based sizing instead of character-based (default: False)
            embedding_model: Embedding model name for token-based sizing (default: 'openai')
            context_optimization: Whether to add section headers and paper title to chunks (default: True)
        """
        self.target_chunk_size = target_chunk_size
        self.base_overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_quality_score = min_quality_score
        self.preserve_abstract = preserve_abstract
        self.adaptive_overlap = adaptive_overlap
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_semantic = use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE and use_semantic
        self.use_tokens = use_tokens and TIKTOKEN_AVAILABLE
        self.embedding_model = embedding_model
        self.context_optimization = context_optimization
        
        # Model-specific token limits and optimization settings
        # Based on 2024-2025 best practices for embedding models
        self.model_token_limits = {
            # OpenAI Models
            'openai': {
                'target': 500, 'max': 800, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'text-embedding-ada-002 (legacy)'
            },
            'openai-3': {
                'target': 512, 'max': 800, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'text-embedding-3-* (recommended: 256-512 tokens)'
            },
            'openai-3-large': {
                'target': 512, 'max': 8191, 'overlap': 100,
                'tokenizer': 'cl100k_base',
                'description': 'text-embedding-3-large (up to 8191 tokens)'
            },
            
            # Cohere Models
            'cohere': {
                'target': 400, 'max': 512, 'overlap': 40,
                'tokenizer': 'cl100k_base',  # Approximate
                'description': 'Cohere embed-english-v3.0 (max 512 tokens)'
            },
            'cohere-light': {
                'target': 300, 'max': 512, 'overlap': 30,
                'tokenizer': 'cl100k_base',
                'description': 'Cohere embed-english-light-v3.0'
            },
            'cohere-multilingual': {
                'target': 400, 'max': 512, 'overlap': 40,
                'tokenizer': 'cl100k_base',
                'description': 'Cohere embed-multilingual-v3.0'
            },
            
            # Sentence-BERT / Sentence-Transformers
            'sentence-bert': {
                'target': 128, 'max': 256, 'overlap': 20,
                'tokenizer': 'gpt2',  # Approximate for BERT
                'description': 'Sentence-BERT (all-MiniLM-L6-v2, all-mpnet-base-v2)'
            },
            'sentence-bert-large': {
                'target': 256, 'max': 512, 'overlap': 25,
                'tokenizer': 'gpt2',
                'description': 'Sentence-BERT large models (up to 512 tokens)'
            },
            
            # Universal Sentence Encoder
            'universal': {
                'target': 512, 'max': 1024, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'Universal Sentence Encoder'
            },
            
            # Long-context Models
            'jina': {
                'target': 1024, 'max': 8192, 'overlap': 100,
                'tokenizer': 'cl100k_base',
                'description': 'Jina Embeddings 2 (long-context, up to 8192 tokens)'
            },
            'voyage': {
                'target': 512, 'max': 4096, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'Voyage AI embeddings (up to 4096 tokens)'
            },
            
            # RAG-Optimized for LLMs (e.g., Mixtral 8x7b, Llama, etc.)
            'mixtral-rag': {
                'target': 512, 'max': 1024, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'RAG-optimized for Mixtral 8x7b (32K context, 512-1024 token chunks recommended)'
            },
            'llm-rag': {
                'target': 512, 'max': 1024, 'overlap': 50,
                'tokenizer': 'cl100k_base',
                'description': 'RAG-optimized for LLMs with large context windows (512-1024 tokens)'
            },
            
            # Default/Generic
            'default': {
                'target': 200, 'max': 400, 'overlap': 20,
                'tokenizer': 'cl100k_base',
                'description': 'Generic fallback for unknown models'
            }
        }
        
        # Initialize tokenizer if using tokens
        self.tokenizer = None
        if self.use_tokens:
            try:
                # Get model-specific configuration
                if embedding_model in self.model_token_limits:
                    limits = self.model_token_limits[embedding_model]
                    tokenizer_name = limits.get('tokenizer', 'cl100k_base')
                    self.target_chunk_size = limits['target']
                    self.max_chunk_size = limits['max']
                    self.base_overlap_size = limits['overlap']
                    model_description = limits.get('description', embedding_model)
                else:
                    # Use default token limits
                    limits = self.model_token_limits['default']
                    tokenizer_name = limits.get('tokenizer', 'cl100k_base')
                    self.target_chunk_size = limits['target']
                    self.max_chunk_size = limits['max']
                    self.base_overlap_size = limits['overlap']
                    model_description = f"Unknown model (using default settings)"
                
                # Initialize appropriate tokenizer
                self.tokenizer = tiktoken.get_encoding(tokenizer_name)
                
                # Store model description for reporting
                self.model_description = model_description
                
            except Exception as e:
                print(f"Warning: Could not initialize tokenizer: {e}")
                print("Falling back to character-based chunking.")
                self.use_tokens = False
                self.model_description = None
        
        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                # Keep NER enabled for entity extraction, disable parser for performance
                self.nlp.disable_pipes(["parser"])
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                # Ensure NER is enabled
                if "ner" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("ner")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found.")
                print("Falling back to regex-based sentence detection.")
                self.use_spacy = False
        
        # Initialize sentence transformer for semantic coherence (optional)
        if self.use_semantic:
            try:
                print("Loading sentence transformer for semantic coherence...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Sentence transformer loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                print("Continuing without semantic coherence feature.")
                self.use_semantic = False
        
        # Major section keywords (IMRaD format)
        self.major_sections = [
            'introduction', 'background', 'related work', 'literature review',
            'methodology', 'methods', 'approach', 'experimental setup',
            'results', 'findings', 'evaluation', 'experiments',
            'discussion', 'analysis', 'interpretation',
            'conclusion', 'conclusions', 'summary', 'future work',
            'references', 'acknowledgments', 'appendix'
        ]
        
        # Section importance weights (for metadata)
        self.section_weights = {
            'abstract': 1.0,
            'introduction': 0.9,
            'background': 0.85,
            'methods': 0.8,
            'methodology': 0.8,
            'results': 0.85,
            'discussion': 0.8,
            'conclusion': 0.9,
            'references': 0.3,
            'acknowledgments': 0.2
        }
        
        # Statistics
        self.stats = {
            'papers_processed': 0,
            'papers_skipped': 0,
            'chunks_created': 0,
            'chunks_improved': 0,
            'section_boundaries_respected': 0,
            'citations_preserved': 0,
            'abstracts_preserved': 0,
            'low_quality_filtered': 0,
            'semantic_groups_created': 0,
            'paragraphs_respected': 0,
            'entities_extracted': 0,
            'table_references_found': 0,
            'figure_references_found': 0,
            'duplicates_removed': 0,
            'total_tokens': 0,
            'context_headers_added': 0,
            'titles_added': 0,
            'errors': []
        }
    
    def detect_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect sentence boundaries in text."""
        if self.use_spacy:
            return self._detect_sentences_spacy(text)
        else:
            return self._detect_sentences_regex(text)
    
    def _detect_sentences_spacy(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect sentences using spaCy."""
        sentences = []
        doc = self.nlp(text)
        for sent in doc.sents:
            start = sent.start_char
            end = sent.end_char
            sentence_text = text[start:end].strip()
            if sentence_text:
                sentences.append((start, end, sentence_text))
        return sentences
    
    def _detect_sentences_regex(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect sentences using regex (fallback method)."""
        sentences = []
        pattern = r'([.!?]+)\s+([A-Z])|([.!?]+)$'
        last_end = 0
        for match in re.finditer(pattern, text):
            end_pos = match.end()
            if self._is_valid_sentence_end(text, match.start(), end_pos):
                sentence_text = text[last_end:end_pos].strip()
                if sentence_text:
                    sentences.append((last_end, end_pos, sentence_text))
                last_end = end_pos
        if last_end < len(text):
            sentence_text = text[last_end:].strip()
            if sentence_text:
                sentences.append((last_end, len(text), sentence_text))
        return sentences
    
    def _is_valid_sentence_end(self, text: str, match_start: int, match_end: int) -> bool:
        """Check if a potential sentence end is valid (not an abbreviation)."""
        abbrev_patterns = [
            r'\b(Dr|Mr|Mrs|Ms|Prof|etc|e\.g|i\.e|vs|Fig|Eq|Ref|Sec|Ch|Vol|No|pp|pp\.|al|et)\s*\.',
            r'\b\d+\.\d+',  # Decimal numbers
            r'\b[A-Z]\.',  # Single letter abbreviations
        ]
        context = text[max(0, match_start-20):match_start+5]
        for pattern in abbrev_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return False
        return True
    
    def is_major_section(self, section_name: str) -> bool:
        """Check if a section name indicates a major section."""
        section_lower = section_name.lower().strip()
        return any(major in section_lower for major in self.major_sections)
    
    def find_section_at_position(self, position: int, sections: List[Dict]) -> Dict:
        """Find which section a character position belongs to."""
        for section in sections:
            start = section.get('start_char', 0)
            end = section.get('end_char', len(section.get('text', '')))
            if start <= position < end:
                return section
        return sections[-1] if sections else {'name': 'Unknown', 'page': 1}
    
    def extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from text."""
        citations = []
        # Pattern: [1], [1,2], [1-3], etc.
        pattern = r'\[(\d+(?:[,\s-]+\d+)*)\]'
        for match in re.finditer(pattern, text):
            citations.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0),
                'numbers': re.findall(r'\d+', match.group(1))
            })
        return citations
    
    def assess_chunk_quality(self, text: str) -> Tuple[float, List[str]]:
        """
        Assess chunk quality and return score (0-1) and issues.
        
        Returns:
            (quality_score, issues_list)
        """
        issues = []
        score = 1.0
        
        # Check for excessive citations
        citations = self.extract_citations(text)
        citation_ratio = len(''.join(c['text'] for c in citations)) / len(text) if text else 0
        if citation_ratio > 0.3:
            issues.append('high_citation_ratio')
            score -= 0.2
        
        # Check for excessive formatting artifacts
        formatting_chars = text.count('\n') + text.count('\t')
        if text and formatting_chars / len(text) > 0.1:
            issues.append('excessive_formatting')
            score -= 0.1
        
        # Check for very short chunks
        if len(text) < 100:
            issues.append('too_short')
            score -= 0.3
        
        # Check for mostly equations/formulas
        math_patterns = r'\$[^$]+\$|\\[a-zA-Z]+|\\begin\{|\\end\{'
        words = text.split()
        if words:
            math_ratio = len(re.findall(math_patterns, text)) / len(words)
            if math_ratio > 0.2:
                issues.append('mostly_math')
                score -= 0.15
        
        # Check for meaningful content
        if len(words) < 20:
            issues.append('insufficient_content')
            score -= 0.2
        
        return max(0.0, score), issues
    
    def get_section_importance(self, section_name: str) -> float:
        """Get importance weight for a section."""
        section_lower = section_name.lower().strip()
        for key, weight in self.section_weights.items():
            if key in section_lower:
                return weight
        return 0.7  # Default weight
    
    def is_abstract_section(self, section_name: str) -> bool:
        """Check if section is an abstract."""
        section_lower = section_name.lower().strip()
        return 'abstract' in section_lower
    
    def group_sentences_semantically(self, sentences: List[Tuple[int, int, str]], 
                                    full_text: str) -> List[List[int]]:
        """
        Group sentences by semantic similarity (optional feature).
        Returns list of sentence index groups.
        """
        if not self.use_semantic or len(sentences) < 3:
            # Return individual sentences if semantic grouping not available
            return [[i] for i in range(len(sentences))]
        
        try:
            # Extract sentence texts
            sentence_texts = [full_text[start:end] for start, end, _ in sentences]
            
            # Get embeddings
            embeddings = self.sentence_model.encode(sentence_texts, show_progress_bar=False)
            
            # Simple clustering: group consecutive sentences with high similarity
            groups = []
            current_group = [0]
            
            for i in range(1, len(embeddings)):
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
                
                if similarity > 0.7:  # High similarity threshold
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]
            
            if current_group:
                groups.append(current_group)
            
            self.stats['semantic_groups_created'] += len(groups)
            return groups
            
        except Exception as e:
            print(f"Warning: Semantic grouping failed: {e}")
            return [[i] for i in range(len(sentences))]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE:
            # Fallback: simple dot product approximation
            if len(vec1) != len(vec2):
                return 0.0
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        else:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
    
    def calculate_adaptive_overlap(self, is_section_boundary: bool, 
                                  is_major_section: bool) -> int:
        """Calculate adaptive overlap size based on context."""
        if not self.adaptive_overlap:
            return self.base_overlap_size
        
        if is_section_boundary and is_major_section:
            # Larger overlap at major section boundaries
            return int(self.base_overlap_size * 1.5)  # 150 chars
        elif is_section_boundary:
            # Medium overlap at minor section boundaries
            return int(self.base_overlap_size * 1.2)  # 120 chars
        else:
            # Standard overlap within sections
            return self.base_overlap_size
    
    def detect_paragraphs(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect paragraph boundaries in text.
        Returns list of (start, end) character positions for each paragraph.
        """
        paragraphs = []
        # Paragraph boundaries: double newlines or newline followed by section-like text
        # Pattern: \n\n or \n\s*\n
        pattern = r'\n\s*\n'
        last_end = 0
        
        for match in re.finditer(pattern, text):
            para_end = match.start()
            if para_end > last_end:
                paragraphs.append((last_end, para_end))
            last_end = match.end()
        
        # Add final paragraph
        if last_end < len(text):
            paragraphs.append((last_end, len(text)))
        
        # If no paragraphs found, treat entire text as one paragraph
        if not paragraphs:
            paragraphs = [(0, len(text))]
        
        return paragraphs
    
    def find_paragraph_at_position(self, position: int, paragraphs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find which paragraph a character position belongs to."""
        for para_start, para_end in paragraphs:
            if para_start <= position < para_end:
                return (para_start, para_end)
        return None
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using spaCy NER.
        Returns list of entity dictionaries with text, label, and position.
        """
        if not self.use_spacy or "ner" not in self.nlp.pipe_names:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                try:
                    description = spacy.explain(ent.label_) or ent.label_
                except:
                    description = ent.label_
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': description
                })
            return entities
        except Exception as e:
            # Fallback if NER fails
            return []
    
    def extract_table_figure_references(self, text: str) -> Dict[str, List[str]]:
        """
        Extract table and figure references from text.
        Returns dict with 'tables' and 'figures' lists.
        """
        references = {'tables': [], 'figures': []}
        
        # Pattern for table references: "Table 1", "Table I", "Tab. 1", etc.
        table_patterns = [
            r'Table\s+([IVX\d]+)',  # Table 1, Table I, Table IV
            r'Tab\.\s*([IVX\d]+)',   # Tab. 1
            r'TABLE\s+([IVX\d]+)',   # TABLE 1
        ]
        
        # Pattern for figure references: "Figure 1", "Fig. 1", "Fig 1", etc.
        figure_patterns = [
            r'Figure\s+([IVX\d]+)',  # Figure 1, Figure I
            r'Fig\.\s*([IVX\d]+)',   # Fig. 1
            r'Fig\s+([IVX\d]+)',     # Fig 1
            r'FIGURE\s+([IVX\d]+)',  # FIGURE 1
        ]
        
        # Extract table references
        for pattern in table_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = match.group(0).strip()
                if ref not in references['tables']:
                    references['tables'].append(ref)
        
        # Extract figure references
        for pattern in figure_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = match.group(0).strip()
                if ref not in references['figures']:
                    references['figures'].append(ref)
        
        return references
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity.
        Returns value between 0 and 1.
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # If texts are identical
        if text1 == text2:
            return 1.0
        
        # If either is empty
        if not text1 or not text2:
            return 0.0
        
        # Use word-based Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def detect_duplicates(self, chunks: List[Dict], similarity_threshold: float = 0.9) -> List[int]:
        """
        Detect duplicate chunks based on text similarity.
        Returns list of chunk indices to remove (keeping the first occurrence).
        
        Args:
            chunks: List of chunk dictionaries
            similarity_threshold: Similarity threshold for considering chunks duplicates (default: 0.9)
        
        Returns:
            List of indices to remove
        """
        indices_to_remove = []
        
        for i in range(len(chunks)):
            if i in indices_to_remove:
                continue
            
            text1 = chunks[i]['text']
            quality1 = chunks[i]['metadata'].get('quality_score', 0.5)
            
            for j in range(i + 1, len(chunks)):
                if j in indices_to_remove:
                    continue
                
                text2 = chunks[j]['text']
                quality2 = chunks[j]['metadata'].get('quality_score', 0.5)
                
                # Calculate similarity
                similarity = self.calculate_text_similarity(text1, text2)
                
                if similarity >= similarity_threshold:
                    # Keep the chunk with higher quality, remove the other
                    if quality1 >= quality2:
                        indices_to_remove.append(j)
                    else:
                        indices_to_remove.append(i)
                        break  # Break inner loop if we're removing current chunk
        
        return sorted(set(indices_to_remove))
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured tokenizer.
        Falls back to character-based estimation if tokenizer not available.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens (or character estimate if tokenizer unavailable)
        """
        if not self.use_tokens or not self.tokenizer:
            # Fallback: approximate 1 token ≈ 4 characters for English
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to character estimate
            return len(text) // 4
    
    def get_section_header_text(self, section: Dict) -> str:
        """
        Format section header for inclusion in chunk text.
        
        Args:
            section: Section dictionary with 'name' and optionally 'text'
        
        Returns:
            Formatted section header string
        """
        section_name = section.get('name', 'Unknown')
        if not section_name or section_name == 'Unknown':
            return ""
        
        # Format: "## Section Name"
        return f"## {section_name}\n\n"
    
    def get_paper_title(self, metadata: Dict) -> str:
        """
        Extract paper title from metadata.
        
        Args:
            metadata: Paper metadata dictionary
        
        Returns:
            Paper title string, or empty string if not found
        """
        # Try multiple possible fields for title
        title = (
            metadata.get('title') or
            metadata.get('Title') or
            metadata.get('paper_title') or
            metadata.get('name') or
            ""
        )
        
        # Clean up title
        if title:
            title = title.strip()
            # Remove extra whitespace
            title = ' '.join(title.split())
        
        return title
    
    def add_context_to_chunk_text(self, 
                                   chunk_text: str,
                                   start_section: Dict,
                                   metadata: Dict,
                                   chunk_index: int,
                                   is_first_chunk: bool = False) -> str:
        """
        Add context (section headers and/or paper title) to chunk text.
        
        Args:
            chunk_text: Original chunk text
            start_section: Section dictionary for the chunk
            metadata: Paper metadata
            chunk_index: Index of the chunk
            is_first_chunk: Whether this is the first chunk of the paper
        
        Returns:
            Chunk text with context added
        """
        if not self.context_optimization:
            return chunk_text
        
        context_parts = []
        
        # Add paper title to first chunk
        if is_first_chunk:
            title = self.get_paper_title(metadata)
            if title:
                context_parts.append(f"# {title}\n\n")
                self.stats['titles_added'] += 1
        
        # Add section header
        section_header = self.get_section_header_text(start_section)
        if section_header:
            context_parts.append(section_header)
            self.stats['context_headers_added'] += 1
        
        # Combine context with chunk text
        if context_parts:
            context_text = ''.join(context_parts)
            return context_text + chunk_text
        
        return chunk_text
    
    def get_chunk_size(self, text: str) -> int:
        """
        Get chunk size (tokens or characters) based on configuration.
        
        Args:
            text: Text to measure
        
        Returns:
            Size in tokens (if use_tokens) or characters (otherwise)
        """
        if self.use_tokens:
            return self.count_tokens(text)
        return len(text)
    
    def normalize_text_for_embeddings(self, text: str) -> str:
        """
        Normalize text for optimal embedding generation.
        Cleans whitespace, normalizes unicode, removes formatting artifacts.
        
        Args:
            text: Raw text to normalize
        
        Returns:
            Normalized text optimized for embeddings
        """
        if not text:
            return text
        
        # Normalize unicode (NFD to NFC)
        try:
            import unicodedata
            text = unicodedata.normalize('NFC', text)
        except ImportError:
            pass  # Skip if unicodedata not available
        
        # Normalize whitespace: multiple spaces -> single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize newlines: multiple newlines -> double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        
        # Normalize quotes (smart quotes -> regular quotes)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes (em-dash, en-dash -> hyphen for consistency)
        # But preserve in citations like [1-3]
        text = re.sub(r'(?<!\[)\u2013(?![0-9])', '-', text)  # en-dash
        text = re.sub(r'(?<!\[)\u2014(?![0-9])', '-', text)  # em-dash
        
        # Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Final strip
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        Uses simple TF-IDF-like approach with noun phrases.
        """
        if not self.use_spacy:
            # Fallback: extract capitalized words and important terms
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            # Filter common words
            common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
            keywords = [w for w in words if w not in common_words]
            return list(set(keywords))[:max_keywords]
        
        try:
            doc = self.nlp(text)
            # Extract important terms
            keywords = []
            
            # Get important nouns and proper nouns (works without parser)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                    if len(token.text) > 3:  # Filter very short words
                        keywords.append(token.text.lower())
            
            # Also extract multi-word terms (simple pattern matching)
            # Look for patterns like "machine learning", "neural network"
            multi_word_pattern = r'\b([a-z]+(?:\s+[a-z]+){1,2})\b'
            for match in re.finditer(multi_word_pattern, text.lower()):
                phrase = match.group(1)
                # Filter out common phrases
                if len(phrase.split()) <= 3 and phrase not in ['the', 'and', 'for', 'with', 'from']:
                    keywords.append(phrase)
            
            # Remove duplicates and return top keywords
            unique_keywords = list(set(keywords))
            # Sort by length (longer = more specific) then return top
            unique_keywords.sort(key=len, reverse=True)
            return unique_keywords[:max_keywords]
        except Exception as e:
            return []
    
    def create_enhanced_chunks(self, 
                              full_text: str,
                              sections: List[Dict],
                              metadata: Dict) -> List[Dict]:
        """
        Create enhanced chunks with all improvements.
        
        Args:
            full_text: Complete text of the paper
            sections: List of section dictionaries
            metadata: Paper metadata
        
        Returns:
            List of enhanced chunk dictionaries
        """
        # Handle empty text
        if not full_text or not full_text.strip():
            return []
        
        # Detect all sentences
        sentences = self.detect_sentences(full_text)
        
        if not sentences:
            # If no sentences detected, create a single chunk from the text if it's large enough
            if len(full_text.strip()) >= self.min_chunk_size:
                return [self._create_enhanced_chunk(
                full_text, 0, len(full_text), sections, metadata, 0, full_text
            )]
        
        # Detect paragraphs for paragraph-aware chunking
        paragraphs = self.detect_paragraphs(full_text)
        
        # Group sentences semantically if enabled
        semantic_groups = None
        if self.use_semantic:
            semantic_groups = self.group_sentences_semantically(sentences, full_text)
        
        # Handle abstract separately if preserve_abstract is enabled
        abstract_chunk = None
        if self.preserve_abstract:
            abstract_text = metadata.get('abstract', '')
            if abstract_text and len(abstract_text) > 50:
                # Find abstract section
                for section in sections:
                    if self.is_abstract_section(section.get('name', '')):
                        section_text = section.get('text', '')
                        if section_text:
                            abstract_chunk = self._create_enhanced_chunk(
                                section_text,
                                section.get('start_char', 0),
                                section.get('end_char', len(section_text)),
                                sections,
                                metadata,
                                0,
                                section_text,
                                is_abstract=True
                            )
                            break
        
        chunks = []
        chunk_index = 0 if abstract_chunk is None else 1
        
        # Skip abstract sentences if we preserved it
        start_sentence_idx = 0
        if abstract_chunk:
            # Find where abstract ends in sentences
            abstract_end = abstract_chunk['metadata']['char_end']
            for i, (start, end, _) in enumerate(sentences):
                if start >= abstract_end:
                    start_sentence_idx = i
                    break
        
        i = start_sentence_idx
        max_iterations = len(sentences) * 2
        iterations = 0
        
        while i < len(sentences) and iterations < max_iterations:
            iterations += 1
            
            # Determine start position with adaptive overlap
            if chunk_index > 0 and chunks:
                prev_chunk = chunks[-1]
                prev_section = prev_chunk['metadata'].get('section', '')
                prev_is_major = self.is_major_section(prev_section)
                prev_section_end = prev_chunk['metadata'].get('section_end', '')
                section_changed = prev_section != prev_section_end
                
                # Calculate adaptive overlap
                overlap_size = self.calculate_adaptive_overlap(
                    section_changed, prev_is_major
                )
                
                prev_chunk_end = prev_chunk['metadata']['char_end']
                
                if self.use_tokens:
                    # For token-based: find overlap by counting tokens backwards
                    overlap_target_pos = prev_chunk_end
                    overlap_tokens = 0
                    chunk_start_sentence_idx = i
                    
                    # Go backwards from current position to find overlap
                    for j in range(i - 1, max(0, i - 20), -1):
                        sent_start, sent_end, _ = sentences[j]
                        overlap_text = full_text[sent_start:prev_chunk_end]
                        overlap_tokens = self.count_tokens(overlap_text)
                        
                        if overlap_tokens >= overlap_size:
                            chunk_start_sentence_idx = j
                            break
                    
                    chunk_start_pos = sentences[chunk_start_sentence_idx][0]
                    i = chunk_start_sentence_idx
                else:
                    # Character-based overlap (original logic)
                    overlap_target = prev_chunk_end - overlap_size
                    
                    # Find sentence that starts at or before overlap_target
                    chunk_start_sentence_idx = i
                    for j in range(i - 1, max(0, i - 20), -1):
                        if sentences[j][0] <= overlap_target:
                            chunk_start_sentence_idx = j
                            break
                    
                    chunk_start_pos = sentences[chunk_start_sentence_idx][0]
                    i = chunk_start_sentence_idx
            else:
                chunk_start_pos = sentences[i][0]
            
            # Find current section
            current_section = self.find_section_at_position(chunk_start_pos, sections)
            current_section_name = current_section.get('name', 'Unknown')
            is_major = self.is_major_section(current_section_name)
            
            # Find current paragraph
            current_para = self.find_paragraph_at_position(chunk_start_pos, paragraphs)
            
            # Collect sentences for this chunk
            chunk_sentences = []
            chunk_end_pos = chunk_start_pos
            section_changed = False
            paragraph_respected = True
            
            # If using semantic groups, iterate over groups instead of individual sentences
            if semantic_groups:
                # Find which semantic group contains sentence i
                current_group_idx = None
                for g_idx, group in enumerate(semantic_groups):
                    if i in group:
                        current_group_idx = g_idx
                        break
                
                if current_group_idx is not None:
                    # Process semantic groups
                    for g_idx in range(current_group_idx, len(semantic_groups)):
                        group = semantic_groups[g_idx]
                        # Get all sentences in this semantic group
                        group_sentences = [sentences[sent_idx] for sent_idx in group if sent_idx < len(sentences)]
                        
                        if not group_sentences:
                            continue
                        
                        # Check if adding this group would exceed max size
                        first_sent_start = group_sentences[0][0]
                        last_sent_end = group_sentences[-1][1]
                        potential_text = full_text[chunk_start_pos:last_sent_end]
                        if self.use_tokens:
                            potential_size = self.count_tokens(potential_text)
                        else:
                            potential_size = last_sent_end - chunk_start_pos
                        
                        # Check section boundary
                        next_section = self.find_section_at_position(first_sent_start, sections)
                        next_section_name = next_section.get('name', 'Unknown')
                        
                        if (is_major and 
                            next_section_name != current_section_name and
                            self.is_major_section(next_section_name)):
                            section_changed = True
                            break
                        
                        # Check paragraph boundary - prefer not splitting paragraphs
                        next_para = self.find_paragraph_at_position(first_sent_start, paragraphs)
                        if current_para and next_para and current_para != next_para:
                            # Would cross paragraph boundary
                            # Only allow if we haven't added any sentences yet or if we're past target size
                            if chunk_sentences and potential_size < self.target_chunk_size:
                                # Try to respect paragraph boundary
                                paragraph_respected = False
                                # But allow if we're still too small
                                if potential_size < self.min_chunk_size:
                                    # Too small, allow crossing paragraph
                                    pass
                                else:
                                    # Respect paragraph boundary
                                    break
                        
                        if potential_size > self.max_chunk_size:
                            break
                        
                        # Add all sentences in this semantic group
                        chunk_sentences.extend(group_sentences)
                        chunk_end_pos = last_sent_end
                        
                        if potential_size >= self.target_chunk_size:
                            # Update i to continue from next group
                            if g_idx + 1 < len(semantic_groups):
                                next_group = semantic_groups[g_idx + 1]
                                if next_group:
                                    i = next_group[0]
                                else:
                                    i = group[-1] + 1
                            else:
                                i = group[-1] + 1
                            break
                    
                    # Update i for next iteration if we processed groups
                    if chunk_sentences:
                        # Find the last sentence index we processed
                        last_processed_group_idx = current_group_idx
                        for g_idx in range(current_group_idx, len(semantic_groups)):
                            group = semantic_groups[g_idx]
                            if any(sentences[sent_idx] in chunk_sentences for sent_idx in group if sent_idx < len(sentences)):
                                last_processed_group_idx = g_idx
                        
                        # Move to next group after the last processed one
                        if last_processed_group_idx + 1 < len(semantic_groups):
                            next_group = semantic_groups[last_processed_group_idx + 1]
                            if next_group and next_group[0] < len(sentences):
                                i = next_group[0]
                            else:
                                i = len(sentences)  # End of sentences
                        else:
                            i = len(sentences)  # No more groups
                else:
                    # Fallback to sentence-by-sentence if semantic group not found
                    semantic_groups = None  # Disable semantic for this iteration
                    # Continue with sentence-by-sentence processing below
            else:
                # Standard sentence-by-sentence processing (or fallback from semantic)
                pass
            
            # Standard sentence-by-sentence processing (if not using semantic groups or as fallback)
            if not semantic_groups:
                while i < len(sentences):
                    sent_start, sent_end, sent_text = sentences[i]
                    
                    # Calculate potential size (tokens or characters)
                    potential_text = full_text[chunk_start_pos:sent_end]
                    if self.use_tokens:
                        potential_size = self.count_tokens(potential_text)
                    else:
                        potential_size = sent_end - chunk_start_pos
                    
                    # Check if next sentence would cross major section boundary
                    next_section = self.find_section_at_position(sent_start, sections)
                    next_section_name = next_section.get('name', 'Unknown')
                    
                    if (is_major and 
                        next_section_name != current_section_name and
                        self.is_major_section(next_section_name)):
                        # Don't cross major section boundary
                        section_changed = True
                        break
                    
                    # Check paragraph boundary - prefer not splitting paragraphs
                    next_para = self.find_paragraph_at_position(sent_start, paragraphs)
                    if current_para and next_para and current_para != next_para:
                        # Would cross paragraph boundary
                        if chunk_sentences and potential_size < self.target_chunk_size:
                            # Try to respect paragraph boundary
                            paragraph_respected = False
                            # But allow if we're still too small
                            if potential_size < self.min_chunk_size:
                                # Too small, allow crossing paragraph
                                pass
                            else:
                                # Respect paragraph boundary
                                break
                    
                    if chunk_sentences and potential_size > self.max_chunk_size:
                        break
                    
                    chunk_sentences.append(sentences[i])
                    chunk_end_pos = sent_end
                    i += 1
                    
                    if potential_size >= self.target_chunk_size:
                        if i < len(sentences):
                            # Check next sentence
                            next_sent_end = sentences[i][1]
                            next_potential_text = full_text[chunk_start_pos:next_sent_end]
                            if self.use_tokens:
                                next_potential_size = self.count_tokens(next_potential_text)
                            else:
                                next_potential_size = next_sent_end - chunk_start_pos
                            
                            if next_potential_size > self.max_chunk_size:
                                break
                        else:
                            break
            
            if not chunk_sentences:
                break
            
            # Create chunk text
            chunk_text = full_text[chunk_start_pos:chunk_end_pos].strip()
            
            # Normalize text for embeddings
            chunk_text = self.normalize_text_for_embeddings(chunk_text)
            
            # Find section for context (before adding context)
            start_section = self.find_section_at_position(chunk_start_pos, sections)
            
            # Store original chunk text for quality assessment (before context)
            original_chunk_text = chunk_text
            
            # Add context (section headers and/or paper title) to chunk text
            is_first_chunk = (chunk_index == 0 and not abstract_chunk)
            chunk_text_with_context = self.add_context_to_chunk_text(
                chunk_text,
                start_section,
                metadata,
                chunk_index,
                is_first_chunk=is_first_chunk
            )
            
            # Use context-enhanced text for the chunk
            chunk_text = chunk_text_with_context
            
            # Track tokens if using token-based sizing (count with context)
            if self.use_tokens:
                token_count = self.count_tokens(chunk_text)
                self.stats['total_tokens'] += token_count
            
            # Quality check (use original text without context for quality assessment)
            quality_score, quality_issues = self.assess_chunk_quality(original_chunk_text)
            if quality_score < self.min_quality_score:
                self.stats['low_quality_filtered'] += 1
                # Try to extend chunk if too short
                if 'too_short' in quality_issues and i < len(sentences):
                    continue
                else:
                    continue
            
            # Find sections (already found above for context, but need end_section)
            if 'start_section' not in locals():
                start_section = self.find_section_at_position(chunk_start_pos, sections)
            end_section = self.find_section_at_position(chunk_end_pos - 1, sections)
            
            # Extract citations (use original text for extraction)
            citations = self.extract_citations(original_chunk_text)
            
            # Extract entities and keywords (use original text for extraction)
            entities = self.extract_entities(original_chunk_text)
            keywords = self.extract_keywords(original_chunk_text)
            
            # Extract table/figure references (use original text for extraction)
            table_figure_refs = self.extract_table_figure_references(original_chunk_text)
            
            if entities:
                self.stats['entities_extracted'] += len(entities)
            
            if table_figure_refs['tables']:
                self.stats['table_references_found'] += len(table_figure_refs['tables'])
            
            if table_figure_refs['figures']:
                self.stats['figure_references_found'] += len(table_figure_refs['figures'])
            
            if paragraph_respected:
                self.stats['paragraphs_respected'] += 1
            
            # Create enhanced chunk (use context-enhanced text for final chunk)
            chunk = self._create_enhanced_chunk(
                chunk_text,  # This now includes context (title + section header)
                chunk_start_pos,
                chunk_end_pos,
                sections,
                metadata,
                chunk_index,
                full_text,
                start_section=start_section,
                end_section=end_section,
                citations=citations,
                quality_score=quality_score,
                quality_issues=quality_issues,
                section_changed=section_changed,
                entities=entities,
                keywords=keywords,
                paragraph_respected=paragraph_respected,
                table_figure_refs=table_figure_refs
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            if section_changed:
                self.stats['section_boundaries_respected'] += 1
            
            if citations:
                self.stats['citations_preserved'] += len(citations)
        
        # Add abstract chunk at the beginning if preserved
        if abstract_chunk:
            chunks.insert(0, abstract_chunk)
            self.stats['abstracts_preserved'] += 1
            # Note: If abstract is preserved, it becomes chunk 0, so first regular chunk won't get title
            # We'll add title to abstract chunk instead if it exists
            if self.context_optimization:
                title = self.get_paper_title(metadata)
                if title and abstract_chunk:
                    abstract_text = abstract_chunk.get('text', '')
                    if not abstract_text.startswith(f"# {title}"):
                        abstract_chunk['text'] = f"# {title}\n\n{abstract_text}"
                        self.stats['titles_added'] += 1
        
        # Deduplication: Remove duplicate chunks
        if len(chunks) > 1:
            duplicate_indices = self.detect_duplicates(chunks, similarity_threshold=0.9)
            if duplicate_indices:
                # Remove duplicates in reverse order to maintain indices
                for idx in reversed(duplicate_indices):
                    chunks.pop(idx)
                self.stats['duplicates_removed'] += len(duplicate_indices)
                # Re-index chunks after removal
                for i, chunk in enumerate(chunks):
                    chunk['metadata']['chunk_index'] = i
                    chunk['chunk_id'] = f"{metadata.get('paper_id', 'unknown')}_chunk_{i}"
        
        self.stats['chunks_created'] += len(chunks)
        return chunks
    
    def _create_enhanced_chunk(self,
                               text: str,
                               char_start: int,
                               char_end: int,
                               sections: List[Dict],
                               metadata: Dict,
                               chunk_index: int,
                               full_text: str,
                               start_section: Optional[Dict] = None,
                               end_section: Optional[Dict] = None,
                               citations: Optional[List[Dict]] = None,
                               quality_score: Optional[float] = None,
                               quality_issues: Optional[List[str]] = None,
                               section_changed: bool = False,
                               is_abstract: bool = False,
                               entities: Optional[List[Dict]] = None,
                               keywords: Optional[List[str]] = None,
                               paragraph_respected: bool = True,
                               table_figure_refs: Optional[Dict] = None) -> Dict:
        """Create an enhanced chunk with comprehensive metadata."""
        paper_id = metadata.get('paper_id', 'unknown')
        
        # Find sections if not provided
        if start_section is None:
            start_section = self.find_section_at_position(char_start, sections)
        if end_section is None:
            end_section = self.find_section_at_position(char_end - 1, sections)
        
        start_section_name = start_section.get('name', 'Unknown')
        end_section_name = end_section.get('name', 'Unknown')
        
        # Extract citations if not provided
        if citations is None:
            citations = self.extract_citations(text)
        
        # Assess quality if not provided
        if quality_score is None or quality_issues is None:
            quality_score, quality_issues = self.assess_chunk_quality(text)
        
        # Get section importance
        section_importance = self.get_section_importance(start_section_name)
        is_major_section = self.is_major_section(start_section_name)
        
        # Extract entities and keywords if not provided
        if entities is None:
            entities = self.extract_entities(text)
        if keywords is None:
            keywords = self.extract_keywords(text)
        
        # Extract table/figure references if not provided
        if table_figure_refs is None:
            table_figure_refs = self.extract_table_figure_references(text)
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for ent in entities:
            entities_by_type[ent['label']].append(ent['text'])
        
        # Calculate token count
        token_count = None
        if self.use_tokens:
            token_count = self.count_tokens(text)
        
        # Enhanced metadata
        chunk_metadata = {
                'paper_id': paper_id,
                'extraction_method': metadata.get('extraction_method', 'unknown'),
                'chunk_index': chunk_index,
                'chunk_method': 'enhanced_sentence_aware' + ('_semantic' if self.use_semantic else '') + ('_token_based' if self.use_tokens else ''),
                'char_start': char_start,
                'char_end': char_end,
                'chunk_length': len(text),
                'token_count': token_count if token_count else None,
                'sizing_method': 'tokens' if self.use_tokens else 'characters',
            'section': start_section_name,
            'page': start_section.get('page', 1),
            'section_end': end_section_name,
            'page_end': end_section.get('page', 1),
            # Enhanced metadata
            'quality_score': round(quality_score, 3),
            'quality_issues': quality_issues,
            'citation_count': len(citations),
            'citations': [c['text'] for c in citations],
            'is_major_section': is_major_section,
            'section_importance': round(section_importance, 2),
            'section_boundary_respected': section_changed or chunk_index == 0,
            'is_abstract': is_abstract,
            # Additional context
            'has_citations': len(citations) > 0,
            'word_count': len(text.split()),
            # New improvements
            'paragraph_respected': paragraph_respected,
            'entity_count': len(entities),
            'entities': [ent['text'] for ent in entities],
            'entities_by_type': dict(entities_by_type),
            'keywords': keywords[:10],  # Limit to top 10 keywords
            # Table/Figure references
            'table_references': table_figure_refs['tables'],
            'figure_references': table_figure_refs['figures'],
            'has_tables': len(table_figure_refs['tables']) > 0,
            'has_figures': len(table_figure_refs['figures']) > 0,
            'table_figure_count': len(table_figure_refs['tables']) + len(table_figure_refs['figures']),
        }
        
        return {
            'chunk_id': f"{paper_id}_chunk_{chunk_index}",
            'text': text,
            'metadata': chunk_metadata
        }
    
    def improve_paper_chunks(self, json_file: Path, output_dir: Optional[Path] = None) -> bool:
        """Improve chunks for a single paper JSON file."""
        try:
            # Calculate output path first to check if already processed
            if output_dir:
                # Preserve directory structure relative to input directory
                try:
                    if hasattr(self, '_input_dir') and self._input_dir:
                        # Calculate relative path from input directory
                        input_dir_resolved = Path(self._input_dir).resolve()
                        json_file_resolved = json_file.resolve()
                        try:
                            relative_path = json_file_resolved.relative_to(input_dir_resolved)
                            output_path = output_dir / relative_path
                        except ValueError:
                            # If not relative, try to extract structure from path
                            # Find common parent and preserve structure
                            parts = json_file.parts
                            # Find where 'output' appears in path
                            if 'output' in parts:
                                output_idx = parts.index('output')
                                # Take everything after 'output'
                                relative_parts = parts[output_idx + 1:]
                                output_path = output_dir / Path(*relative_parts)
                            else:
                                # Fallback: preserve parent directory
                                output_path = output_dir / json_file.parent.name / json_file.name
                    else:
                        # Fallback: preserve parent directory structure
                        output_path = output_dir / json_file.parent.name / json_file.name
                except Exception:
                    # Final fallback: use filename only
                    output_path = output_dir / json_file.name
                
                # Skip if already processed
                if output_path.exists():
                    self.stats['papers_skipped'] = self.stats.get('papers_skipped', 0) + 1
                    return True  # Return True because file is already done
            else:
                output_path = json_file
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paper_id = data.get('paper_id', 'unknown')
            full_text = data.get('text', {}).get('full', '')
            
            if not full_text:
                self.stats['errors'].append(f"{paper_id}: No full text found")
                return False
            
            # Extract sections
            sections_data = data.get('text', {}).get('sections', [])
            sections = []
            for section in sections_data:
                sections.append({
                    'name': section.get('name', 'Unknown'),
                    'start_char': section.get('start_char', 0),
                    'end_char': section.get('end_char', len(section.get('text', ''))),
                    'page': section.get('page', 1),
                    'text': section.get('text', '')
                })
            
            if not sections:
                sections = [{
                    'name': 'Full Text',
                    'start_char': 0,
                    'end_char': len(full_text),
                    'page': 1,
                    'text': full_text
                }]
            
            # Get paper metadata
            paper_metadata = data.get('metadata', {})
            
            # Get old chunk count BEFORE replacing (for statistics)
            old_chunk_count = len(data.get('chunks', []))
            
            # Create enhanced chunks
            new_chunks = self.create_enhanced_chunks(
                full_text,
                sections,
                paper_metadata
            )
            
            # Update data with new chunks
            data['chunks'] = new_chunks
            
            # Update statistics
            if 'statistics' in data:
                data['statistics']['num_chunks'] = len(new_chunks)
                data['statistics']['chunking_method'] = 'enhanced_sentence_aware'
                data['statistics']['avg_chunk_size'] = (
                    sum(len(c['text']) for c in new_chunks) / len(new_chunks)
                    if new_chunks else 0
                )
            
            # Save improved data (output_path already calculated above)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.stats['papers_processed'] += 1
            self.stats['chunks_improved'] += old_chunk_count
            
            return True
            
        except Exception as e:
            error_msg = f"{json_file}: {str(e)}"
            self.stats['errors'].append(error_msg)
            print(f"Error processing {json_file}: {e}")
            return False
    
    def process_directory(self, 
                         input_dir: Path,
                         output_dir: Optional[Path] = None,
                         max_files: Optional[int] = None,
                         verbose: bool = True,
                         num_workers: int = None):
        """Process all JSON files in a directory."""
        # Store input directory for path preservation
        self._input_dir = input_dir.resolve()
        
        all_json_files = list(input_dir.rglob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.')
        ]
        
        if max_files:
            json_files = json_files[:max_files]
        
        total = len(json_files)
        if verbose:
            print(f"Processing {total} files...")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
        
        if num_workers == 1 or total == 1:
            # Sequential processing for single worker or single file
            for i, json_file in enumerate(json_files, 1):
                if verbose and i % 100 == 0:
                    print(f"Processed {i}/{total} files... "
                          f"({self.stats['chunks_created']} chunks created)")
                
                self.improve_paper_chunks(json_file, output_dir)
        else:
            # Parallel processing
            if verbose:
                print(f"Using {num_workers} workers for parallel processing...")
            
            # Create worker function with configuration
            worker_func = partial(
                _process_single_file_worker,
                target_chunk_size=self.target_chunk_size,
                overlap_size=self.base_overlap_size,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size,
                use_spacy=self.use_spacy,
                use_semantic=self.use_semantic,
                min_quality_score=self.min_quality_score,
                preserve_abstract=self.preserve_abstract,
                adaptive_overlap=self.adaptive_overlap,
                use_tokens=self.use_tokens,
                embedding_model=self.embedding_model,
                context_optimization=self.context_optimization,
                output_dir=output_dir,
                input_dir=input_dir
            )
            
            # Process files in parallel
            with mp.Pool(processes=num_workers) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(worker_func, json_files), 1):
                    results.append(result)
                    if verbose and i % 100 == 0:
                        processed = sum(1 for r in results if r.get('success', False))
                        chunks = sum(r.get('chunks_created', 0) for r in results)
                        print(f"Processed {i}/{total} files... "
                              f"({processed} successful, {chunks} chunks created)")
                
                # Aggregate statistics
                for result in results:
                    if result.get('success', False):
                        self.stats['papers_processed'] += 1
                        self.stats['papers_skipped'] += result.get('papers_skipped', 0)
                        self.stats['chunks_created'] += result.get('chunks_created', 0)
                        self.stats['chunks_improved'] += result.get('chunks_improved', 0)
                        self.stats['section_boundaries_respected'] += result.get('section_boundaries', 0)
                        self.stats['paragraphs_respected'] += result.get('paragraphs_respected', 0)
                        self.stats['citations_preserved'] += result.get('citations_preserved', 0)
                        self.stats['entities_extracted'] += result.get('entities_extracted', 0)
                        self.stats['table_references_found'] += result.get('table_refs', 0)
                        self.stats['figure_references_found'] += result.get('figure_refs', 0)
                        self.stats['duplicates_removed'] += result.get('duplicates_removed', 0)
                        self.stats['context_headers_added'] += result.get('context_headers', 0)
                        self.stats['titles_added'] += result.get('titles_added', 0)
                        self.stats['abstracts_preserved'] += result.get('abstracts_preserved', 0)
                        self.stats['low_quality_filtered'] += result.get('low_quality_filtered', 0)
                        self.stats['total_tokens'] += result.get('total_tokens', 0)
                    else:
                        self.stats['errors'].append(result.get('error', 'Unknown error'))
        
        if verbose:
            self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*80)
        print("ENHANCED CHUNK PROCESSING STATISTICS")
        print("="*80)
        print(f"Papers processed: {self.stats['papers_processed']}")
        if self.stats.get('papers_skipped', 0) > 0:
            print(f"Papers skipped (already processed): {self.stats['papers_skipped']}")
        print(f"Chunks created: {self.stats['chunks_created']}")
        print(f"Old chunks replaced: {self.stats['chunks_improved']}")
        print(f"Section boundaries respected: {self.stats['section_boundaries_respected']}")
        print(f"Paragraphs respected: {self.stats['paragraphs_respected']}")
        print(f"Citations preserved: {self.stats['citations_preserved']}")
        print(f"Entities extracted: {self.stats['entities_extracted']}")
        print(f"Table references found: {self.stats['table_references_found']}")
        print(f"Figure references found: {self.stats['figure_references_found']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        if self.context_optimization:
            print(f"Context headers added: {self.stats['context_headers_added']}")
            print(f"Paper titles added: {self.stats['titles_added']}")
        if self.use_tokens and self.stats['total_tokens'] > 0:
            print(f"Total tokens: {self.stats['total_tokens']:,}")
            if self.stats['chunks_created'] > 0:
                avg_tokens = self.stats['total_tokens'] / self.stats['chunks_created']
                print(f"Average tokens per chunk: {avg_tokens:.1f}")
        print(f"Abstracts preserved: {self.stats['abstracts_preserved']}")
        print(f"Low quality chunks filtered: {self.stats['low_quality_filtered']}")
        if self.use_semantic:
            print(f"Semantic groups created: {self.stats['semantic_groups_created']}")
        print(f"Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nFirst 10 errors:")
            for error in self.stats['errors'][:10]:
                print(f"  - {error}")


def _process_single_file_worker(json_file: Path,
                                 target_chunk_size: int,
                                 overlap_size: int,
                                 min_chunk_size: int,
                                 max_chunk_size: int,
                                 use_spacy: bool,
                                 use_semantic: bool,
                                 min_quality_score: float,
                                 preserve_abstract: bool,
                                 adaptive_overlap: bool,
                                 use_tokens: bool,
                                 embedding_model: str,
                                 context_optimization: bool,
                                 output_dir: Optional[Path],
                                 input_dir: Optional[Path] = None) -> Dict:
    """
    Worker function for parallel processing.
    Creates a new improver instance for each worker (required for spaCy).
    """
    try:
        # Create improver instance for this worker
        improver = AdvancedChunkImprover(
            target_chunk_size=target_chunk_size,
            overlap_size=overlap_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            use_spacy=use_spacy,
            use_semantic=use_semantic,
            min_quality_score=min_quality_score,
            preserve_abstract=preserve_abstract,
            adaptive_overlap=adaptive_overlap,
            use_tokens=use_tokens,
            embedding_model=embedding_model,
            context_optimization=context_optimization
        )
        
        # Store input directory for path preservation
        if input_dir:
            improver._input_dir = input_dir.resolve()
        
        # Process the file
        success = improver.improve_paper_chunks(json_file, output_dir)
        
        if success:
            return {
                'success': True,
                'papers_skipped': improver.stats.get('papers_skipped', 0),
                'chunks_created': improver.stats['chunks_created'],
                'chunks_improved': improver.stats['chunks_improved'],
                'section_boundaries': improver.stats['section_boundaries_respected'],
                'paragraphs_respected': improver.stats['paragraphs_respected'],
                'citations_preserved': improver.stats['citations_preserved'],
                'entities_extracted': improver.stats['entities_extracted'],
                'table_refs': improver.stats['table_references_found'],
                'figure_refs': improver.stats['figure_references_found'],
                'duplicates_removed': improver.stats['duplicates_removed'],
                'context_headers': improver.stats['context_headers_added'],
                'titles_added': improver.stats['titles_added'],
                'abstracts_preserved': improver.stats['abstracts_preserved'],
                'low_quality_filtered': improver.stats['low_quality_filtered'],
                'total_tokens': improver.stats['total_tokens']
            }
        else:
            return {
                'success': False,
                'error': f"{json_file}: Processing failed"
            }
    except Exception as e:
        return {
            'success': False,
            'error': f"{json_file}: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive chunk quality improvement with all enhancements'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: overwrite input files)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=950,
        help='Target chunk size in characters or tokens (default: 950)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=100,
        help='Base overlap size in characters or tokens (default: 100)'
    )
    parser.add_argument(
        '--use-tokens',
        action='store_true',
        help='Use token-based chunk sizing instead of character-based'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='openai',
        choices=[
            'openai', 'openai-3', 'openai-3-large',
            'cohere', 'cohere-light', 'cohere-multilingual',
            'sentence-bert', 'sentence-bert-large',
            'universal', 'jina', 'voyage',
            'mixtral-rag', 'llm-rag', 'default'
        ],
        help='Embedding model for token-based sizing (default: openai). '
             'Options: openai, openai-3, openai-3-large, cohere, cohere-light, '
             'cohere-multilingual, sentence-bert, sentence-bert-large, universal, jina, voyage, default'
    )
    parser.add_argument(
        '--no-context-optimization',
        action='store_true',
        help='Disable context window optimization (section headers and paper title)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    parser.add_argument(
        '--no-spacy',
        action='store_true',
        help='Disable spaCy (use regex-based sentence detection)'
    )
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='Enable semantic coherence grouping (requires sentence-transformers)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.5,
        help='Minimum quality score to keep chunk (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--no-abstract-preserve',
        action='store_true',
        help='Disable abstract preservation'
    )
    parser.add_argument(
        '--no-adaptive-overlap',
        action='store_true',
        help='Disable adaptive overlap'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 10 files'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1, use 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    
    if args.test:
        args.max_files = 10
        print("Running in TEST mode (10 files only)")
    
    improver = AdvancedChunkImprover(
        target_chunk_size=args.target_size,
        overlap_size=args.overlap,
        use_spacy=not args.no_spacy,
        use_semantic=args.semantic,
        min_quality_score=args.min_quality,
        preserve_abstract=not args.no_abstract_preserve,
        adaptive_overlap=not args.no_adaptive_overlap,
        use_tokens=args.use_tokens,
        embedding_model=args.embedding_model,
        context_optimization=not args.no_context_optimization
    )
    
    print("="*80)
    print("COMPREHENSIVE CHUNK QUALITY IMPROVEMENT")
    print("="*80)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path or '(overwrite input)'}")
    sizing_unit = "tokens" if improver.use_tokens else "chars"
    print(f"Target chunk size: {improver.target_chunk_size} {sizing_unit}")
    print(f"Base overlap size: {improver.base_overlap_size} {sizing_unit}")
    print(f"Sizing method: {'Token-based' if improver.use_tokens else 'Character-based'}")
    if improver.use_tokens:
        print(f"Embedding model: {improver.embedding_model}")
        if hasattr(improver, 'model_description') and improver.model_description:
            print(f"Model description: {improver.model_description}")
    print(f"Sentence detection: {'spaCy' if improver.use_spacy else 'regex'}")
    print(f"Semantic coherence: {'Enabled' if improver.use_semantic else 'Disabled'}")
    print(f"Abstract preservation: {'Enabled' if improver.preserve_abstract else 'Disabled'}")
    print(f"Adaptive overlap: {'Enabled' if improver.adaptive_overlap else 'Disabled'}")
    print(f"Context optimization: {'Enabled' if improver.context_optimization else 'Disabled'}")
    print(f"Quality filtering: Enabled (min score: {args.min_quality})")
    print("="*80)
    print("Features enabled:")
    print("  ✓ Sentence-aware chunking")
    print("  ✓ Section-aware chunking (IMRaD boundaries)")
    print("  ✓ Paragraph-aware chunking")
    print("  ✓ Citation preservation")
    print("  ✓ Quality filtering")
    print("  ✓ Enhanced metadata")
    print("  ✓ Named Entity Recognition (NER)")
    print("  ✓ Keyword extraction")
    print("  ✓ Table/Figure reference detection")
    print("  ✓ Deduplication")
    print("  ✓ Text normalization for embeddings")
    if improver.use_tokens:
        print(f"  ✓ Token-based chunk sizing ({improver.embedding_model})")
    if improver.context_optimization:
        print("  ✓ Context window optimization (section headers + paper title)")
    if improver.preserve_abstract:
        print("  ✓ Abstract preservation")
    if improver.adaptive_overlap:
        print("  ✓ Adaptive overlap")
    if improver.use_semantic:
        print("  ✓ Semantic coherence grouping (integrated)")
    print("="*80)
    
    improver.process_directory(
        input_path,
        output_path,
        max_files=args.max_files,
        verbose=True,
        num_workers=args.workers
    )
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
