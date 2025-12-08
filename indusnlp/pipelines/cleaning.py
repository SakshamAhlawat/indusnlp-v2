"""
Text Cleaning Pipeline - Core cleaning logic for Hindi/Indic text.
Refactored from test_cleaning_automated.py
"""

import os
import re
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from filters.textcleaner import TextCleaner
from filters.HindiTextCleaner import HindiTextCleaner
from filters.badwords_en_hi_hiR import badword_list


class CleaningPipeline:
    """
    Master cleaning pipeline for Hindi/Indic text.
    Handles LaTeX protection, table preservation, bad word filtering, and transliteration.
    """
    
    def __init__(self, transliterate: bool = True, filter_badwords: bool = True):
        """
        Initialize the cleaning pipeline.
        
        Args:
            transliterate: Enable English to Hindi transliteration
            filter_badwords: Enable bad word filtering
        """
        self.transliterate = transliterate
        self.filter_badwords = filter_badwords
        
        # Initialize cleaners
        config = [
            ("handle_whitespace", None),
            ("remove_redundant_lines", None),
            ("remove_blank_lines", None),
        ]
        self.textcleaner = TextCleaner(config)
        self.hicleaner = HindiTextCleaner(transliterate=transliterate)
        
        # Load bad words set
        self.bad_words_set = self._load_bad_words()
    
    def _load_bad_words(self) -> set:
        """Load bad words from the badword_list."""
        b_set = set()
        for w in badword_list:
            if len(w.strip()) > 1:
                b_set.add(w.strip().lower())
        return b_set
    
    def check_bad_word(self, text: str) -> bool:
        """
        Check if text contains any bad word phrase.
        
        Args:
            text: Text to check
            
        Returns:
            True if bad word found, False otherwise
        """
        if not self.filter_badwords or not self.bad_words_set:
            return False
        
        text_lower = text.lower()
        for bad_word in self.bad_words_set:
            if bad_word in text_lower:
                return True
        return False

    def mask_bad_words(self, text: str) -> str:
        """Mask bad words in-place instead of dropping the entire line.

        Each bad word occurrence is replaced with asterisks of the same length,
        preserving surrounding text.
        """
        if not self.filter_badwords or not self.bad_words_set or not text:
            return text

        lower = text.lower()
        for bad_word in self.bad_words_set:
            start = 0
            bw_len = len(bad_word)
            if bw_len == 0:
                continue

            while True:
                idx = lower.find(bad_word, start)
                if idx == -1:
                    break

                end = idx + bw_len
                # Replace in the original text; keep length same
                text = text[:idx] + ("*" * bw_len) + text[end:]
                lower = text.lower()
                start = end

        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using the master cleaning pipeline.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # --- STEP 1: PROTECT LATEX (STRICTLY SINGLE-LINE) ---
        latex_map = {}
        latex_counter = [0]
        
        def protect_latex(match):
            placeholder = f"__LTX_{latex_counter[0]}__"
            latex_map[placeholder] = match.group(0)
            latex_counter[0] += 1
            return placeholder
        
        text = re.sub(r'(`[^\n]*?`)', protect_latex, text)
        text = re.sub(r'(\$\$[^\n]*?\$\$)', protect_latex, text)
        text = re.sub(r'(\$[^\n\$]+\$)', protect_latex, text)
        
        # --- STEP 2: SMART LINE-BY-LINE PROCESSING ---
        lines = text.split('\n')
        final_cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # 2a. Identify Table lines
            is_separator = re.match(r'^[\s\|\-\:]+$', stripped) and '|' in stripped and '-' in stripped
            is_table_row = stripped.startswith('|') and stripped.endswith('|')
            
            # 2b. Identify LaTeX lines
            has_latex = "__LTX_" in line
            
            if is_table_row or is_separator:
                # TABLE: Keep as-is
                final_cleaned_lines.append(line)
            
            elif has_latex:
                # LATEX: Safe clean only
                cleaned_line = line.replace('$', '')
                cleaned_line = re.sub(r'\([a-zA-Z]+\)', '', cleaned_line)
                cleaned_line = self.textcleaner(cleaned_line)
                final_cleaned_lines.append(cleaned_line)
            
            else:
                # NORMAL TEXT

                # Mask bad words instead of dropping the line
                line = self.mask_bad_words(line)

                # Normal Cleaning
                cleaned_line = line.replace('$', '')
                cleaned_line = re.sub(r'\([a-zA-Z]+\)', '', cleaned_line)
                cleaned_line = self.textcleaner(cleaned_line)
                
                # Hindi cleaner
                cleaned_line = self.hicleaner(cleaned_line)
                
                if cleaned_line and cleaned_line.strip():
                    final_cleaned_lines.append(cleaned_line)
        
        text = '\n'.join(final_cleaned_lines)
        
        # --- STEP 3: RESTORE LATEX ---
        for placeholder, original_latex in latex_map.items():
            text = text.replace(placeholder, original_latex)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a single text file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned = self.clean_text(content)
            
            output_file = output_path / input_path.name.replace(".txt", "_cleaned.txt")
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def process_files(self, input_path: Path, output_path: Path) -> dict:
        """
        Process files from input path (file, directory, or ZIP).
        
        Args:
            input_path: Path to input (file/dir/zip)
            output_path: Path to output directory
            
        Returns:
            Dict with processing results
        """
        results = {"processed": 0, "failed": 0, "files": []}
        temp_dirs = []
        
        try:
            jobs = self._gather_jobs(input_path, output_path, temp_dirs)
            
            for source, dest in jobs:
                if self.process_file(source, dest):
                    results["processed"] += 1
                    results["files"].append(str(source.name))
                else:
                    results["failed"] += 1
        finally:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return results
    
    def _gather_jobs(self, input_path: Path, output_path: Path, temp_dirs: list) -> list:
        """Gather all text files to process."""
        input_path = input_path.resolve()
        output_path = output_path.resolve()
        jobs = []
        
        if not input_path.exists():
            return jobs
        
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == ".txt":
                jobs.append((input_path, output_path))
            elif suffix == ".zip":
                zip_stem = input_path.stem
                extraction_root = Path(tempfile.mkdtemp(prefix="clean_zip_"))
                temp_dirs.append(extraction_root)
                target_dir = extraction_root / zip_stem
                target_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(target_dir)
                
                zip_output = output_path / zip_stem
                zip_output.mkdir(parents=True, exist_ok=True)
                
                for txt in sorted(target_dir.rglob("*.txt")):
                    if not txt.name.endswith("_cleaned.txt"):
                        jobs.append((txt, zip_output))
        
        elif input_path.is_dir():
            for txt in sorted(input_path.rglob("*.txt")):
                if not txt.name.endswith("_cleaned.txt"):
                    jobs.append((txt, output_path))
        
        return jobs


# Module-level function for backward compatibility
_default_pipeline = None

def get_pipeline(transliterate: bool = True, filter_badwords: bool = True) -> CleaningPipeline:
    """Get or create a cleaning pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = CleaningPipeline(transliterate=transliterate, filter_badwords=filter_badwords)
    return _default_pipeline


def master_cleaning_pipeline(text: str, transliterate: bool = True, filter_badwords: bool = True) -> str:
    """
    Clean text using the master cleaning pipeline.
    
    Args:
        text: Raw text to clean
        transliterate: Enable transliteration
        filter_badwords: Enable bad word filtering
        
    Returns:
        Cleaned text
    """
    pipeline = CleaningPipeline(transliterate=transliterate, filter_badwords=filter_badwords)
    return pipeline.clean_text(text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IndusNLP cleaning pipeline.")
    parser.add_argument("--input", required=True, help="Path to input file/folder/zip")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-transliterate", action="store_true", help="Disable transliteration")
    parser.add_argument("--no-filter-badwords", action="store_true", help="Disable bad word filtering")
    args = parser.parse_args()
    
    pipeline = CleaningPipeline(
        transliterate=not args.no_transliterate,
        filter_badwords=not args.no_filter_badwords
    )
    
    results = pipeline.process_files(Path(args.input), Path(args.output))
    print(f"✅ Processed: {results['processed']}, Failed: {results['failed']}")
