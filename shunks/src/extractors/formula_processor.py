"""
Formula and Mathematical Expression Processing Module
Handles detection, normalization, and preservation of mathematical formulas.
"""

import re
from typing import List, Dict, Tuple, Optional
from loguru import logger


class FormulaProcessor:
    """Process and normalize mathematical formulas and expressions."""
    
    def __init__(self, preserve_latex: bool = True, normalize_spacing: bool = True):
        self.preserve_latex = preserve_latex
        self.normalize_spacing = normalize_spacing
        
        # Common Greek letters mapping
        self.greek_letters = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho',
            'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
            'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
            'Θ': 'Theta', 'Λ': 'Lambda', 'Π': 'Pi', 'Σ': 'Sigma',
            'Φ': 'Phi', 'Ω': 'Omega'
        }
        
        # Mathematical operators and symbols
        self.math_operators = {
            '∑': 'sum', '∏': 'product', '∫': 'integral', '√': 'sqrt',
            '±': 'pm', '×': 'times', '÷': 'div', '≤': 'le', '≥': 'ge',
            '≠': 'ne', '≈': 'approx', '∞': 'infty', '∂': 'partial',
            '∇': 'nabla', '∆': 'Delta', '∈': 'in', '∉': 'notin',
            '⊂': 'subset', '⊃': 'superset', '∪': 'cup', '∩': 'cap'
        }
    
    def detect_formulas(self, text: str) -> List[Dict[str, any]]:
        """
        Detect mathematical formulas and expressions in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected formulas with positions and types
        """
        formulas = []
        
        # Pattern 1: LaTeX display math: $$...$$ or \[...\]
        latex_display = re.finditer(r'\$\$[^\$]+\$\$|\\\[[^\]]+\\\]', text)
        for match in latex_display:
            formulas.append({
                'type': 'latex_display',
                'content': match.group(),
                'start': match.start(),
                'end': match.end(),
                'formatted': self._format_latex(match.group())
            })
        
        # Pattern 2: LaTeX inline math: $...$
        latex_inline = re.finditer(r'\$[^\$]+\$', text)
        for match in latex_inline:
            formulas.append({
                'type': 'latex_inline',
                'content': match.group(),
                'start': match.start(),
                'end': match.end(),
                'formatted': self._format_latex(match.group())
            })
        
        # Pattern 3: Equations with = sign and mathematical expressions
        # Format: variable = expression
        equation_pattern = r'[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^\n]{5,100}'
        equations = re.finditer(equation_pattern, text)
        for match in equations:
            content = match.group()
            # Check if it looks like a mathematical equation
            if self._is_mathematical(content):
                formulas.append({
                    'type': 'equation',
                    'content': content,
                    'start': match.start(),
                    'end': match.end(),
                    'formatted': self._normalize_formula(content)
                })
        
        # Pattern 4: Standalone formulas (lines with math symbols)
        math_symbols = r'[∑∏∫√±×÷≤≥≠≈∞∂∇∆∈∉⊂⊃∪∩αβγδεζηθικλμνξπρστυφχψω]'
        standalone = re.finditer(rf'^[^\n]*{math_symbols}[^\n]*$', text, re.MULTILINE)
        for match in standalone:
            content = match.group().strip()
            if len(content) > 3 and self._is_mathematical(content):
                formulas.append({
                    'type': 'standalone_formula',
                    'content': content,
                    'start': match.start(),
                    'end': match.end(),
                    'formatted': self._normalize_formula(content)
                })
        
        # Pattern 5: Subscripts and superscripts (common in formulas)
        # Format: x_1, x^2, x_{i}, x^{n}
        sub_sup = re.finditer(r'[A-Za-z0-9]_\s*\{?[0-9A-Za-z]+\}?|[A-Za-z0-9]\^\s*\{?[0-9A-Za-z]+\}?', text)
        for match in sub_sup:
            # Only include if in context of other math
            context = text[max(0, match.start()-20):match.end()+20]
            if self._is_mathematical(context):
                formulas.append({
                    'type': 'subscript_superscript',
                    'content': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'formatted': self._normalize_sub_sup(match.group())
                })
        
        # Sort by position
        formulas.sort(key=lambda x: x['start'])
        
        return formulas
    
    def _is_mathematical(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        # Check for math symbols
        math_symbols = r'[∑∏∫√±×÷≤≥≠≈∞∂∇∆∈∉⊂⊃∪∩αβγδεζηθικλμνξπρστυφχψω\+\-\*/\(\)\^]'
        if re.search(math_symbols, text):
            return True
        
        # Check for common math patterns
        math_patterns = [
            r'[A-Za-z]\s*=\s*[A-Za-z0-9]',  # Simple equations
            r'[A-Za-z]_\d+',  # Subscripts
            r'[A-Za-z]\^\d+',  # Superscripts
            r'\\[a-z]+\{',  # LaTeX commands
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def normalize_formulas(self, text: str) -> str:
        """
        Normalize mathematical formulas in text.
        
        Args:
            text: Text containing formulas
            
        Returns:
            Text with normalized formulas
        """
        formulas = self.detect_formulas(text)
        
        # Process from end to start to preserve indices
        for formula in reversed(formulas):
            start = formula['start']
            end = formula['end']
            original = formula['content']
            normalized = formula.get('formatted', original)
            
            # Replace with normalized version
            text = text[:start] + normalized + text[end:]
        
        return text
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize a single formula."""
        # Remove excessive whitespace
        if self.normalize_spacing:
            # Normalize spaces around operators
            formula = re.sub(r'\s*([+\-*/=<>≤≥≠≈])\s*', r' \1 ', formula)
            # Normalize spaces around parentheses
            formula = re.sub(r'\s*([()])\s*', r'\1', formula)
            # Remove multiple spaces
            formula = re.sub(r'\s+', ' ', formula)
        
        # Fix common formatting issues
        # Fix: "β=" -> "β ="
        formula = re.sub(r'([α-ωΑ-Ω])([=<>])', r'\1 \2', formula)
        # Fix: "=β" -> "= β"
        formula = re.sub(r'([=<>])([α-ωΑ-Ω])', r'\1 \2', formula)
        
        # Normalize subscripts: x_1 -> x_1 (ensure proper spacing)
        formula = re.sub(r'([A-Za-z0-9])\s*_\s*(\d+)', r'\1_\2', formula)
        # Normalize superscripts: x^2 -> x^2
        formula = re.sub(r'([A-Za-z0-9])\s*\^\s*(\d+)', r'\1^\2', formula)
        
        return formula.strip()
    
    def _normalize_sub_sup(self, text: str) -> str:
        """Normalize subscripts and superscripts."""
        # Remove spaces around _ and ^
        text = re.sub(r'\s*_\s*', '_', text)
        text = re.sub(r'\s*\^\s*', '^', text)
        # Normalize braces
        text = re.sub(r'_\s*\{', '_{', text)
        text = re.sub(r'\^\s*\{', '^{', text)
        return text
    
    def _format_latex(self, latex: str) -> str:
        """Format LaTeX expression."""
        # Remove $ delimiters if present
        latex = latex.strip('$')
        latex = latex.replace('\\[', '').replace('\\]', '')
        
        if self.preserve_latex:
            # Preserve LaTeX as-is but normalize spacing
            latex = re.sub(r'\s+', ' ', latex)
            return latex.strip()
        else:
            # Could convert to readable format here
            return latex
    
    def extract_formula_context(self, text: str, formula_start: int, context_chars: int = 100) -> str:
        """Extract context around a formula."""
        start = max(0, formula_start - context_chars)
        end = min(len(text), formula_start + context_chars)
        return text[start:end]
    
    def preserve_formulas_in_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Preserve formulas by marking them and returning both processed text and formula list.
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (processed_text, formulas_list)
        """
        formulas = self.detect_formulas(text)
        
        # Replace formulas with placeholders
        processed_text = text
        formula_map = {}
        
        for i, formula in enumerate(reversed(formulas)):
            placeholder = f"__FORMULA_{len(formulas) - i - 1}__"
            start = formula['start']
            end = formula['end']
            
            formula_map[placeholder] = formula
            processed_text = processed_text[:start] + placeholder + processed_text[end:]
        
        return processed_text, list(formula_map.values())
    
    def restore_formulas(self, text: str, formulas: List[Dict]) -> str:
        """Restore formulas from placeholders."""
        for i, formula in enumerate(formulas):
            placeholder = f"__FORMULA_{i}__"
            if placeholder in text:
                # Use formatted version if available
                replacement = formula.get('formatted', formula.get('content', ''))
                text = text.replace(placeholder, replacement)
        
        return text


def improve_formula_formatting(text: str) -> str:
    """
    Improve formatting of mathematical formulas in text.
    This is a convenience function that applies common fixes.
    
    Args:
        text: Text with formulas
        
    Returns:
        Text with improved formula formatting
    """
    processor = FormulaProcessor(preserve_latex=True, normalize_spacing=True)
    
    # Common fixes for badly formatted formulas
    
    # Fix 1: Spacing around operators (but preserve in subscripts/superscripts)
    # Pattern: variable operator variable (but not x_1 or x^2)
    text = re.sub(r'([A-Za-z0-9α-ωΑ-Ω])\s*([+\-*/=<>≤≥≠≈])\s*([A-Za-z0-9α-ωΑ-Ω])', r'\1 \2 \3', text)
    
    # Fix 2: Greek letters spacing with operators (critical fix)
    # Pattern: β= or =β -> β = or = β
    text = re.sub(r'([α-ωΑ-Ω])\s*([=<>≤≥≠≈+\-*/])', r'\1 \2', text)
    text = re.sub(r'([=<>≤≥≠≈+\-*/])\s*([α-ωΑ-Ω])', r'\1 \2', text)
    
    # Fix 3: Spacing around Greek letters in expressions
    # Pattern: whileβis -> while β is
    text = re.sub(r'([a-zA-Z])([α-ωΑ-Ω])', r'\1 \2', text)
    text = re.sub(r'([α-ωΑ-Ω])([a-zA-Z])', r'\1 \2', text)
    
    # Fix 4: Subscripts: x_1 -> x_1 (remove spaces around _)
    text = re.sub(r'([A-Za-z0-9])\s*_\s*(\d+)', r'\1_\2', text)
    text = re.sub(r'([A-Za-z0-9])\s*_\s*\{([^\}]+)\}', r'\1_{\2}', text)
    text = re.sub(r'([A-Za-z0-9])\s*_\s*([A-Za-z])', r'\1_\2', text)  # x_i
    
    # Fix 5: Superscripts: x^2 -> x^2
    text = re.sub(r'([A-Za-z0-9])\s*\^\s*(\d+)', r'\1^\2', text)
    text = re.sub(r'([A-Za-z0-9])\s*\^\s*\{([^\}]+)\}', r'\1^{\2}', text)
    
    # Fix 6: Array brackets and parentheses in formulas
    # Pattern: [t] or (t) should not have spaces inside
    text = re.sub(r'\[\s*([^\]]+)\s*\]', r'[\1]', text)  # [t] not [ t ]
    text = re.sub(r'\(\s*([^\)]+)\s*\)', r'(\1)', text)  # (t) not ( t )
    
    # Fix 7: Fix common patterns like "U[t]" spacing
    text = re.sub(r'([A-Za-z])\s*\[', r'\1[', text)  # U [t] -> U[t]
    text = re.sub(r'\]\s*([=<>+\-*/])', r']\1', text)  # ] = -> ]=
    text = re.sub(r'([=<>+\-*/])\s*\[', r'\1[', text)  # = [ -> =[
    
    # Fix 8: Fix spacing in expressions like "W X[t]" -> "W * X[t]" or "W X[t]"
    # This is tricky - we'll preserve it but ensure consistent spacing
    text = re.sub(r'([A-Za-z])\s+([A-Z])\s*\[', r'\1 \2[', text)  # W X[t] -> W X[t]
    
    # Fix 9: Remove spaces in negative signs that are part of formulas
    # Pattern: "e −∆t" -> "e - ∆t" or "e-∆t" (but be careful with subtraction)
    text = re.sub(r'([a-zA-Z0-9])\s*−\s*([A-Za-z])', r'\1 - \2', text)  # e −∆t -> e - ∆t
    
    # Fix 10: Normalize multiple spaces to single space (but preserve formula structure)
    # Do this carefully to not break formulas
    text = re.sub(r'([^=+\-*/<>])\s{2,}([^=+\-*/<>])', r'\1 \2', text)
    
    # Normalize the formulas using the processor
    text = processor.normalize_formulas(text)
    
    return text

