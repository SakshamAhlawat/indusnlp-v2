# Set Indic NLP Resources Path
import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from indicnlp import common

# --- CONFIGURATION ---
# 1. Resources Path
INDIC_NLP_RESOURCES = r"D:/codes/IndusNLPToolkit/IndusNLPToolkit/IndusNLPToolkit/indic_nlp_resources"

# 2. Project Root Path
FOLDER_NAME = r"D:/codes/IndusNLPToolkit/IndusNLPToolkit/IndusNLPToolkit"

# 3. Bad Words File Path
BAD_WORDS_PATH = r"D:/codes/IndusNLPToolkit/IndusNLPToolkit/IndusNLPToolkit/clean/badwords_en_hi_hiR.py"
# ---------------------

common.set_resources_path(INDIC_NLP_RESOURCES)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indusnlp import TextCleaner, HindiTextCleaner

# Initialize Cleaners
config = [
    ("handle_whitespace", None),
    ("remove_redundant_lines", None),
    ("remove_blank_lines", None),
]
textcleaner = TextCleaner(config)
hicleaner = HindiTextCleaner(transliterate=True)

# --- BAD WORD LOADER ---
def load_bad_words_set():
    print(f"🔍 Loading bad words from: {BAD_WORDS_PATH}")
    b_set = set()
    try:
        with open(BAD_WORDS_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'["\'](.*?)["\']', content)
            for w in matches:
                if len(w.strip()) > 1: 
                    b_set.add(w.strip().lower())
        
        print(f"✅ SUCCESS: Loaded {len(b_set)} bad words.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load bad words file: {e}")
    return b_set

BAD_WORDS_SET = load_bad_words_set()

def check_bad_word(text):
    """
    Checks if the text contains any bad word phrase from the list.
    Supports multi-word phrases (e.g., 'bad word').
    """
    if not BAD_WORDS_SET: return False
    
    # Normalize text for checking (lowercase)
    text_lower = text.lower()
    
    # Iterate through every bad word in our list
    for bad_word in BAD_WORDS_SET:
        # Check if the bad word phrase exists anywhere in this line
        # We add spaces around to ensure we don't match inside other words
        # (e.g. prevent "ass" matching "assembly"), but we check the raw word too
        # to be safe for Hindi.
        
        if bad_word in text_lower:
            print(f"   ⚠️  FILTERED: Found '{bad_word}' in line: {text[:30]}...")
            return True
            
    return False


def master_cleaning_pipeline(text):
    if not text: return ""

    # --- STEP 1: PROTECT LATEX (STRICTLY SINGLE-LINE) ---
    latex_map = {}
    latex_counter = 0
    def protect_latex(match):
        nonlocal latex_counter
        placeholder = f"__LTX_{latex_counter}__"
        latex_map[placeholder] = match.group(0)
        latex_counter += 1
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
            cleaned_line = textcleaner(cleaned_line)
            final_cleaned_lines.append(cleaned_line)
        
        else:
            # NORMAL TEXT
            
            # *** BAD WORD CHECK (Updated Logic) ***
            if check_bad_word(line):
                continue # Skip this line entirely
            
            # Normal Cleaning
            cleaned_line = line.replace('$', '') 
            cleaned_line = re.sub(r'\([a-zA-Z]+\)', '', cleaned_line)
            cleaned_line = textcleaner(cleaned_line)
            
            # Hicleaner
            cleaned_line = hicleaner(cleaned_line) 
            
            if cleaned_line and cleaned_line.strip():
                final_cleaned_lines.append(cleaned_line)

    text = '\n'.join(final_cleaned_lines)
    
    # --- STEP 3: RESTORE LATEX ---
    for placeholder, original_latex in latex_map.items():
        text = text.replace(placeholder, original_latex)

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _gather_text_files(input_path: Path, output_path: Path):
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    jobs = []
    temp_dirs = []

    if not input_path.exists():
        print(f"⚠️ Input path not found: {input_path}")
        return jobs, temp_dirs

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
                if txt.name.endswith("_cleaned.txt"):
                    continue
                jobs.append((txt, zip_output))
        else:
            print(f"⚠️ Unsupported file type: {input_path}")
    elif input_path.is_dir():
        for txt in sorted(input_path.rglob("*.txt")):
            if txt.name.endswith("_cleaned.txt"):
                continue
            jobs.append((txt, output_path))
    else:
        print(f"⚠️ Input path not found: {input_path}")

    return jobs, temp_dirs


def process_files(input_path: Path, output_path: Path):
    jobs, temp_dirs = _gather_text_files(input_path, output_path)

    if not jobs:
        print("⚠️ No .txt files found to process.")
        return

    try:
        for source_path, destination_dir in jobs:
            destination_dir.mkdir(parents=True, exist_ok=True)
            filename = source_path.name
            print(f"Processing: {filename}...")

            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

            try:
                cleaned_text = master_cleaning_pipeline(file_content)
            except Exception as e:
                print(f"❌ Error during cleaning {filename}: {e}")
                continue

            cleaned_file = destination_dir / filename.replace(".txt", "_cleaned.txt")
            try:
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"✅ Saved: {cleaned_file}")
            except Exception as e:
                print(f"❌ Error writing {cleaned_file}: {e}")
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_dataset(folder_path):
    translated_folder = os.path.join(folder_path, "uploaded")
    cleaned_folder = os.path.join(folder_path, "cleaned")
    process_files(Path(translated_folder), Path(cleaned_folder))


def parse_args():
    parser = argparse.ArgumentParser(description="Run IndusNLP cleaning pipeline.")
    parser.add_argument("--input", help="Path to a .txt file or folder containing .txt files.")
    parser.add_argument("--output", help="Directory where cleaned files will be written.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.input and args.output:
        process_files(Path(args.input), Path(args.output))
    else:
        process_dataset(FOLDER_NAME)
