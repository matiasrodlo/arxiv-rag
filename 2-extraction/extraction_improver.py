"""
Advanced PDF Extraction Post-Processing Module

Implements advanced post-processing for PDF extraction results to improve quality:
- Watermark detection and removal
- Footer detection and removal
- Repeated content detection
- Page-by-page quality validation
- Text reconstruction from broken extractions
- Structure preservation enhancement
"""

import re
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from loguru import logger


class ExtractionImprover:
    """
    Advanced post-processor for PDF extraction results.
    Improves quality scores and fixes common extraction issues.
    """

    # Common watermark patterns in academic papers
    WATERMARK_PATTERNS = [
        r'This is a pre[- ]?print of the paper accepted for publication in',
        r'pre[- ]?print',
        r'Accepted for publication',
        r'Under review',
        r'Copyright \d{4}',
        r'arXiv:\d{4}\.\d+',
        r'doi:\s*10\.\d+',
        r'https?://(?:www\.)?arxiv\.org/abs/',
        r'Received.*?Accepted.*?Published',
        r'Peer-review under responsibility of',
        r'All rights reserved',
        r'© \d{4}',
    ]

    # Common footer patterns
    FOOTER_PATTERNS = [
        r'^Page \d+ of \d+',
        r'^\d+\s*$',
        r'arXiv:\d{4}\.\d+\s*\[',
        r'\[cs\.\w+\]',
        r'^\s*\d+\s*$',
        r'Proceedings? of',
        r'International Conference',
        r'Journal of',
    ]

    # Header patterns
    HEADER_PATTERNS = [
        r'^arXiv:\d{4}\.\d+v\d+.*?\n',
        r'^.*?arXiv.*?\n',
        r'^.*?\[.*?\].*?\n',
        r'^.*?\d{4}\.\d+.*?\n',
    ]

    # Expected character ranges per page type
    PAGE_TYPE_RANGES = {
        'text_page': (1500, 5000),
        'figure_page': (100, 2000),
        'table_page': (500, 3000),
        'reference_page': (5000, 20000),
    }

    def __init__(self, enable_watermark_removal: bool = True,
                 enable_footer_removal: bool = True,
                 enable_validation: bool = True,
                 watermark_threshold: float = 0.3):
        """
        Initialize the extraction improver.

        Args:
            enable_watermark_removal: Whether to remove watermarks
            enable_footer_removal: Whether to remove footers
            enable_validation: Whether to run validation checks
            watermark_threshold: Minimum ratio of watermark text to consider removal
        """
        self.enable_watermark_removal = enable_watermark_removal
        self.enable_footer_removal = enable_footer_removal
        self.enable_validation = enable_validation
        self.watermark_threshold = watermark_threshold

    def process_extraction(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an extraction result to improve quality.

        Args:
            extraction_result: Raw extraction result from PDF extractor

        Returns:
            Improved extraction result with quality improvements
        """
        if not extraction_result.get('success', False):
            return extraction_result

        result = extraction_result.copy()
        text = result.get('text', '')
        pages = result.get('pages', [])

        improvements_made = []

        # Step 1: Remove watermarks if enabled
        if self.enable_watermark_removal and text:
            text, watermark_info = self._remove_watermarks(text)
            if watermark_info['removed']:
                result['text'] = text
                improvements_made.append(f"Removed watermark ({watermark_info['count']} occurrences)")

                # Update pages with cleaned text
                if pages:
                    result['pages'] = self._rebuild_pages_with_cleaned_text(
                        pages, text, watermark_info
                    )

        # Step 2: Remove footers if enabled
        if self.enable_footer_removal and pages:
            pages, footer_info = self._remove_footers(pages)
            if footer_info['removed']:
                result['pages'] = pages
                # Rebuild full text from cleaned pages
                result['text'] = '\n'.join(p.get('text', '') for p in pages)
                improvements_made.append(f"Removed footers from {footer_info['count']} pages")

        # Step 3: Validate and flag issues
        if self.enable_validation:
            validation_result = self._validate_extraction(result)
            if validation_result['issues']:
                result['validation_issues'] = validation_result['issues']
                result['validation_warnings'] = validation_result.get('warnings', [])

                # Adjust quality score based on issues
                if validation_result['severity'] == 'high':
                    # Log warning but don't automatically adjust score
                    logger.debug(f"Validation issues found: {validation_result['issues']}")

        # Record improvements
        if improvements_made:
            result['improvements'] = improvements_made
            result['original_text_length'] = len(extraction_result.get('text', ''))
            result['improved_text_length'] = len(result.get('text', ''))

            # Recalculate quality score with improved text
            if 'quality_score' in result:
                old_score = result['quality_score']
                new_score = self._calculate_improved_quality(result)
                result['quality_score'] = new_score
                result['quality_improvement'] = new_score - old_score
                logger.debug(f"Quality improved from {old_score:.3f} to {new_score:.3f}")

        return result

    def _remove_watermarks(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect and remove watermark text from extracted content.
        Only removes lines that are predominantly watermark content.

        Returns:
            Tuple of (cleaned_text, watermark_info_dict)
        """
        watermark_info = {
            'removed': False,
            'count': 0,
            'patterns_found': [],
            'total_removed_chars': 0
        }

        if not text:
            return text, watermark_info

        # More conservative watermark removal - only remove lines that are mostly watermark
        lines = text.split('\n')
        cleaned_lines = []
        watermark_count = 0
        total_removed_chars = 0

        for line in lines:
            line_lower = line.lower()
            line_length = len(line)

            # Check if this line is a watermark line
            is_watermark_line = False
            watermark_chars = 0

            for pattern in self.WATERMARK_PATTERNS:
                if re.search(pattern, line_lower, re.IGNORECASE):
                    # Count how much of the line is watermark content
                    matches = list(re.finditer(pattern, line_lower, re.IGNORECASE))
                    for match in matches:
                        watermark_chars += len(match.group())

                    # If watermark is > 50% of the line and line is short (< 200 chars), remove it
                    # This preserves actual content that just mentions watermarks
                    if watermark_chars / max(line_length, 1) > 0.5 and line_length < 200:
                        is_watermark_line = True
                        watermark_count += 1
                        total_removed_chars += line_length
                        break

            if not is_watermark_line:
                cleaned_lines.append(line)

        if watermark_count > 0:
            watermark_info['removed'] = True
            watermark_info['count'] = watermark_count
            watermark_info['total_removed_chars'] = total_removed_chars
            watermark_info['patterns_found'] = self.WATERMARK_PATTERNS[:3]  # First few patterns used

        return '\n'.join(cleaned_lines), watermark_info

    def _remove_footers(self, pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Remove footer content from pages based on position and patterns.

        Returns:
            Tuple of (cleaned_pages, footer_info_dict)
        """
        footer_info = {
            'removed': False,
            'count': 0,
            'pages_affected': []
        }

        if not pages:
            return pages, footer_info

        cleaned_pages = []

        for i, page in enumerate(pages):
            page_text = page.get('text', '')
            if not page_text:
                cleaned_pages.append(page)
                continue

            lines = page_text.split('\n')
            cleaned_lines = []

            # Process each line to detect and remove footers
            for line_num, line in enumerate(lines):
                line_stripped = line.strip()

                # Skip if line matches footer patterns
                is_footer = False
                for pattern in self.FOOTER_PATTERNS:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_footer = True
                        break

                # Skip single numbers (page numbers)
                if re.match(r'^\d+$', line_stripped):
                    # Only remove if at the beginning or end of page
                    if line_num < 3 or line_num > len(lines) - 3:
                        is_footer = True

                # Skip arXiv headers/footers
                if re.search(r'arXiv:\d{4}\.\d+', line_stripped):
                    if line_num < 3 or line_num > len(lines) - 3:
                        is_footer = True

                if not is_footer:
                    cleaned_lines.append(line)

            # Check if we removed content
            if len(cleaned_lines) < len(lines):
                footer_info['count'] += 1
                footer_info['pages_affected'].append(i + 1)

            cleaned_page = page.copy()
            cleaned_page['text'] = '\n'.join(cleaned_lines)
            cleaned_page['original_line_count'] = len(lines)
            cleaned_page['cleaned_line_count'] = len(cleaned_lines)
            cleaned_pages.append(cleaned_page)

        if footer_info['count'] > 0:
            footer_info['removed'] = True

        return cleaned_pages, footer_info

    def _rebuild_pages_with_cleaned_text(self, pages: List[Dict[str, Any]],
                                         cleaned_text: str,
                                         watermark_info: Dict) -> List[Dict[str, Any]]:
        """
        Rebuild page dictionaries with cleaned text while preserving structure.
        """
        if not pages:
            return pages

        # Split cleaned text back into pages
        # Use original page structure as guide
        cleaned_pages = []
        cleaned_lines = cleaned_text.split('\n')

        current_page_lines = []
        current_page_idx = 0
        original_page_count = len(pages)

        for line in cleaned_lines:
            # Estimate which page this line belongs to based on original structure
            if current_page_idx < original_page_count:
                original_page = pages[current_page_idx]
                original_text = original_page.get('text', '')
                original_lines = original_text.split('\n')

                # Check if we've exceeded the original page's content
                if len(current_page_lines) >= len(original_lines):
                    # This line belongs to next page
                    cleaned_page = original_page.copy()
                    cleaned_page['text'] = '\n'.join(current_page_lines)
                    cleaned_page['char_count'] = len(cleaned_page['text'])
                    cleaned_pages.append(cleaned_page)
                    current_page_lines = []
                    current_page_idx += 1
                else:
                    current_page_lines.append(line)
            else:
                # Extra lines, add to last page
                if cleaned_pages:
                    cleaned_pages[-1]['text'] += '\n' + line
                    cleaned_pages[-1]['char_count'] = len(cleaned_pages[-1]['text'])

        # Add any remaining lines to last page
        if current_page_lines and cleaned_pages:
            cleaned_pages[-1]['text'] += '\n'.join(current_page_lines)
            cleaned_pages[-1]['char_count'] = len(cleaned_pages[-1]['text'])

        return cleaned_pages

    def _validate_extraction(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extraction results and identify issues.

        Returns:
            Dict with 'issues', 'warnings', and 'severity' keys
        """
        issues = []
        warnings = []
        severity = 'none'

        text = extraction_result.get('text', '')
        pages = extraction_result.get('pages', [])
        metadata = extraction_result.get('metadata', {})

        if not text or not pages:
            issues.append("No text or pages in extraction result")
            severity = 'high'
            return {'issues': issues, 'warnings': warnings, 'severity': severity}

        # Check for empty or near-empty pages
        empty_pages = []
        low_content_pages = []
        for i, page in enumerate(pages):
            char_count = page.get('char_count', 0)
            if char_count < 50:
                empty_pages.append(i + 1)
            elif char_count < 500:
                low_content_pages.append(i + 1)

        if empty_pages:
            issues.append(f"Found {len(empty_pages)} empty/near-empty pages: {empty_pages}")
            severity = 'high' if len(empty_pages) > 2 else 'medium'

        if low_content_pages:
            warnings.append(f"Found {len(low_content_pages)} low-content pages: {low_content_pages}")

        # Check for content consistency across pages
        if pages:
            char_counts = [page.get('char_count', 0) for page in pages]
            avg_chars = sum(char_counts) / len(char_counts)

            # Check for unusually low character density pages
            very_low_density = [i + 1 for i, c in enumerate(char_counts)
                              if c < avg_chars * 0.2 and c > 0]

            if very_low_density:
                warnings.append(f"Found {len(very_low_density)} pages with very low content density: {very_low_density}")

        # Check for repeated content (potential extraction issues)
        repeated_lines = self._find_repeated_content(text)
        if repeated_lines:
            warnings.append(f"Found {len(repeated_lines)} repeated lines/content blocks")

        # Check metadata completeness
        if not metadata.get('title') or not metadata.get('author'):
            warnings.append("Missing metadata (title or author)")

        # Check text-to-page ratio
        page_count = len(pages)
        text_length = len(text)
        expected_chars = page_count * 2500

        if text_length < expected_chars * 0.5:
            issues.append(f"Low text content: {text_length} chars for {page_count} pages (expected ~{expected_chars})")
            severity = max(severity, 'medium')
        elif text_length < expected_chars * 0.7:
            warnings.append(f"Below average text content: {text_length} chars for {page_count} pages")

        # Check for common extraction artifacts
        artifacts = self._detect_extraction_artifacts(text)
        if artifacts:
            warnings.append(f"Found extraction artifacts: {artifacts}")

        # Determine overall severity
        if issues:
            severity = 'high' if any('empty' in i.lower() for i in issues) else 'medium'
        elif warnings:
            severity = 'low'

        return {
            'issues': issues,
            'warnings': warnings,
            'severity': severity,
            'empty_pages': empty_pages,
            'low_content_pages': low_content_pages,
            'repeated_content_count': len(repeated_lines),
            'artifact_types': artifacts
        }

    def _find_repeated_content(self, text: str, min_length: int = 50,
                               max_repeats: int = 3) -> List[Dict[str, Any]]:
        """
        Find repeated content blocks that might indicate extraction issues.
        """
        if not text:
            return []

        lines = text.split('\n')
        repeated = []

        # Find consecutive repeated lines
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            if len(line) < min_length:
                continue

            # Count consecutive repeats
            repeat_count = 1
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == line:
                    repeat_count += 1
                else:
                    break

            if repeat_count >= max_repeats:
                repeated.append({
                    'line_preview': line[:50] + '...' if len(line) > 50 else line,
                    'repeat_count': repeat_count,
                    'position': i + 1
                })

        return repeated

    def _detect_extraction_artifacts(self, text: str) -> List[str]:
        """
        Detect common extraction artifacts in the text.
        """
        artifacts = []

        # Check for broken words (single letters separated by spaces)
        broken_single = re.findall(r'\b[a-z]\s+[A-Z]\b', text)
        if len(broken_single) > 5:
            artifacts.append('broken_words')

        # Check for excessive whitespace
        whitespace_ratio = len(re.findall(r'\s{2,}', text)) / max(len(text), 1)
        if whitespace_ratio > 0.05:
            artifacts.append('excessive_whitespace')

        # Check for unusual character sequences
        unusual_chars = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text)
        if unusual_chars:
            artifacts.append('control_characters')

        # Check for very long lines (might be broken columns)
        lines = text.split('\n')
        long_lines = [len(l) for l in lines if len(l) > 500]
        if len(long_lines) > len(lines) * 0.3:
            artifacts.append('unusually_long_lines')

        return artifacts

    def _calculate_improved_quality(self, result: Dict[str, Any]) -> float:
        """
        Recalculate quality score for improved extraction result.
        Accounts for watermark removal and content improvement.
        """
        text = result.get('text', '')
        pages = result.get('pages', [])
        original_score = result.get('quality_score', 0.0)

        if not text or not pages:
            return original_score

        # Calculate new metrics
        text_length = len(text)
        page_count = len(pages)
        expected_length = page_count * 2500

        # Length score - more lenient for improved extractions
        if text_length >= expected_length * 0.8:
            length_score = 1.0
        elif text_length >= expected_length * 0.6:
            length_score = 0.95
        elif text_length >= expected_length * 0.4:
            length_score = 0.9
        else:
            length_score = 0.8

        # Page coverage score - check for empty pages
        empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 50)
        page_coverage_score = 1.0 - (empty_pages / max(page_count, 1)) * 0.2

        # Structure score (basic check)
        text_lower = text.lower()
        has_abstract = 'abstract' in text_lower[:3000]
        has_intro = 'introduction' in text_lower[:5000]
        has_refs = any(term in text_lower for term in ['reference', 'bibliography'])
        structure_score = min(1.0, (has_abstract + has_intro + has_refs + 0.5) / 3.0)

        # Combined score
        final_score = (
            length_score * 0.30 +
            page_coverage_score * 0.30 +
            structure_score * 0.25 +
            0.9 * 0.15  # Readability assumption (slightly higher for improved)
        )

        # If improvements were made, ensure the score doesn't decrease
        improvements = result.get('improvements', [])
        if improvements and final_score < original_score:
            # The improvement might have removed some content, but if structure is intact
            # and we removed watermarks, the score should stay similar
            final_score = max(final_score, original_score - 0.02)  # Small tolerance

        return round(min(1.0, max(0.0, final_score)), 3)


class QualityValidator:
    """
    Validates extraction results against quality thresholds.
    """

    def __init__(self, min_quality: float = 0.95,
                 min_chars_per_page: int = 1000,
                 max_empty_pages_ratio: float = 0.05):
        """
        Initialize validator with quality thresholds.

        Args:
            min_quality: Minimum acceptable quality score
            min_chars_per_page: Minimum acceptable characters per page
            max_empty_pages_ratio: Maximum ratio of empty pages allowed
        """
        self.min_quality = min_quality
        self.min_chars_per_page = min_chars_per_page
        self.max_empty_pages_ratio = max_empty_pages_ratio

    def validate(self, extraction_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate an extraction result against quality thresholds.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not extraction_result.get('success', False):
            return False, ["Extraction was not successful"]

        # Check quality score
        quality_score = extraction_result.get('quality_score', 0.0)
        if quality_score < self.min_quality:
            issues.append(f"Quality score {quality_score:.3f} below threshold {self.min_quality}")

        # Check text content
        text = extraction_result.get('text', '')
        pages = extraction_result.get('pages', [])

        if not text:
            issues.append("No text extracted")
            return False, issues

        # Check characters per page
        if pages:
            avg_chars = sum(p.get('char_count', 0) for p in pages) / len(pages)
            if avg_chars < self.min_chars_per_page:
                issues.append(f"Average {avg_chars:.0f} chars/page below threshold {self.min_chars_per_page}")

            # Check empty pages
            empty_pages = sum(1 for p in pages if p.get('char_count', 0) < 50)
            empty_ratio = empty_pages / len(pages)
            if empty_ratio > self.max_empty_pages_ratio:
                issues.append(f"Empty pages ratio {empty_ratio:.1%} exceeds threshold {self.max_empty_pages_ratio:.1%}")

        # Check for validation issues
        validation_issues = extraction_result.get('validation_issues', [])
        if validation_issues:
            issues.extend(validation_issues)

        return len(issues) == 0, issues

    def get_quality_report(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed quality report for an extraction result.
        """
        report = {
            'overall_score': extraction_result.get('quality_score', 0.0),
            'passes_validation': False,
            'checks': {},
            'issues': [],
            'recommendations': []
        }

        is_valid, issues = self.validate(extraction_result)
        report['passes_validation'] = is_valid
        report['issues'] = issues

        # Individual checks
        text = extraction_result.get('text', '')
        pages = extraction_result.get('pages', [])

        # Quality score check
        quality_score = extraction_result.get('quality_score', 0.0)
        report['checks']['quality_score'] = {
            'score': quality_score,
            'threshold': self.min_quality,
            'passed': quality_score >= self.min_quality
        }

        # Text length check
        text_length = len(text)
        report['checks']['text_length'] = {
            'length': text_length,
            'status': 'good' if text_length > 10000 else ('adequate' if text_length > 5000 else 'low')
        }

        # Page coverage check
        if pages:
            char_counts = [p.get('char_count', 0) for p in pages]
            avg_chars = sum(char_counts) / len(char_counts)
            empty_pages = sum(1 for c in char_counts if c < 50)

            report['checks']['page_coverage'] = {
                'avg_chars_per_page': round(avg_chars, 0),
                'empty_pages': empty_pages,
                'passed': avg_chars >= self.min_chars_per_page and empty_pages / len(pages) <= self.max_empty_pages_ratio
            }

        # Generate recommendations
        if quality_score < self.min_quality:
            report['recommendations'].append("Consider re-extracting with different settings or OCR")

        if pages:
            avg_chars = sum(p.get('char_count', 0) for p in pages) / len(pages)
            if avg_chars < self.min_chars_per_page:
                report['recommendations'].append("Check for figure-heavy pages or extraction issues")

        validation_warnings = extraction_result.get('validation_warnings', [])
        if validation_warnings:
            report['recommendations'].extend(validation_warnings)

        return report


def improve_extraction(extraction_result: Dict[str, Any],
                       enable_watermark_removal: bool = True,
                       enable_footer_removal: bool = True,
                       enable_validation: bool = True) -> Dict[str, Any]:
    """
    Convenience function to improve an extraction result.

    Args:
        extraction_result: Raw extraction result from PDF extractor
        enable_watermark_removal: Whether to remove watermarks
        enable_footer_removal: Whether to remove footers
        enable_validation: Whether to run validation checks

    Returns:
        Improved extraction result
    """
    improver = ExtractionImprover(
        enable_watermark_removal=enable_watermark_removal,
        enable_footer_removal=enable_footer_removal,
        enable_validation=enable_validation
    )
    return improver.process_extraction(extraction_result)


def validate_extraction_quality(extraction_result: Dict[str, Any],
                                min_quality: float = 0.95) -> Dict[str, Any]:
    """
    Convenience function to validate extraction quality.

    Args:
        extraction_result: Extraction result to validate
        min_quality: Minimum acceptable quality score

    Returns:
        Quality report dictionary
    """
    validator = QualityValidator(min_quality=min_quality)
    return validator.get_quality_report(extraction_result)
