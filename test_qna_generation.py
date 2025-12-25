#!/usr/bin/env python3
"""
📂 test_qna_generation.py - Generate 300 Q&A from TXT/ZIP/Folder
🔥 Optimized version with production-grade fixes from QnAPipeline
"""

import argparse
import os
import json
import re
import time
import math
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# 🔑 Configure Gemini (SECURE: no hardcoded key)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not configured. Set it in environment variables.")
    
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ============================================================
# 🧹 FIXED JSON Cleaning (Production-grade)
# ============================================================
def clean_json(raw: str) -> Optional[List[Dict]]:
    """Clean and parse JSON from Gemini response."""
    if not raw:
        return None
    
    # FIXED: Robust regex for ALL code fences
    raw = re.sub(r"```
    start, end = raw.find("["), raw.rfind("]") + 1
    if start == -1 or end <= 0:
        return None
    
    text = raw[start:end].strip()
    
    # FIXED: Smart quote normalization + robust parsing
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("'", "'")
    
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        # Conservative cleanup for trailing commas
        try:
            cleaned = re.sub(r",(\s*[\]}])", r"\1", text)
            data = json.loads(cleaned)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None

# ============================================================
# 🤖 FIXED Batch Generation (Production-grade)
# ============================================================
def generate_detailed_qna(text: str, seen: set) -> Optional[List[Dict]]:
    """Generate a single batch of Q&A pairs."""
    prompt = f"""
You are an expert educational AI that generates *detailed, high-quality academic Q&A*.

🎯 TASK:
Generate 25 **unique** and **non-repetitive** Question–Answer pairs from the given Hindi or bilingual NCERT text.

Distribute evenly among these 5 types:
1️⃣ Multiple Choice Questions (MCQs)
2️⃣ Objective Questions (True/False, Fill in the Blanks, Match the Following)
3️⃣ Summarization Questions
4️⃣ Chain of Thought Questions
5️⃣ Logical Reasoning Questions

⚙️ OUTPUT RULES:
- Return a **valid JSON array only**, no markdown, no text.
- Each object must follow one of these schemas:

🅰️ For MCQs:
{{
  "type": "MCQ",
  "question": "string (Hindi or bilingual)",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "string (exactly one of the options)",
  "explanation": "3–6 sentence explanation"
}}

📘 For other types:
{{
  "type": "Objective" | "Summarization" | "Chain of Thought" | "Logical Reasoning",
  "question": "string",
  "answer": "detailed 4–8 sentence answer"
}}

⚠️ RULES:
- Avoid exact duplicates.
- Semantically similar questions allowed if explanations differ.
- Maintain conceptual variety.
- Output clean JSON only.

Text:
{text[:7000]}
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.5,
                top_p=0.95,
                top_k=40,
            ),
        )
        
        # FIXED: Robust response access
        raw = getattr(response, "text", None)
        if raw is None and hasattr(response, "candidates") and response.candidates:
            try:
                raw = response.candidates.content.parts.text
            except (IndexError, AttributeError):
                raw = ""
        else:
            raw = ""
        
        return clean_json(raw)
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return None

# ============================================================
# 🚀 FIXED Single File Processing (Production-grade)
# ============================================================
def process_file(filepath: str, output_dir: str, num_questions: int = 300, batch_size: int = 25) -> Dict[str, Any]:
    """Process a single text file and generate Q&A."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        raise IOError(f"Failed to read {filepath}: {e}")
    
    if not text:
        raise ValueError(f"Empty file: {filepath}")
    
    # FIXED: Safe chunking
    text_len = len(text)
    chunk_size = min(6000, text_len)
    if chunk_size == 0:
        chunk_size = 1
    
    base_name = os.path.splitext(os.path.basename(filepath))
    os.makedirs(output_dir, exist_ok=True)
    output_txt = os.path.join(output_dir, f"{base_name}_{num_questions}_QA.txt")
    output_json = os.path.join(output_dir, f"{base_name}_{num_questions}_QA.json")
    
    all_qna: List[Dict] = []
    seen: set = set()
    num_batches = math.ceil(num_questions / batch_size)
    
    print(f"\n🧠 Generating {num_questions} questions for: {base_name} ({text_len} chars)")
    
    for batch in range(1, num_batches + 1):
        print(f"⚙️ Batch {batch}/{num_batches} ...")
        
        start = ((batch - 1) * chunk_size) % text_len
        chunk_text = text[start:start + chunk_size]
        
        if not chunk_text.strip():
            print(f"⚠️ Empty chunk, skipping batch {batch}")
            continue
        
        qna_data = None
        for attempt in range(1, 4):  # FIXED: Exponential backoff
            qna_data = generate_detailed_qna(chunk_text, seen)
            if qna_data:
                break
            print(f"⚠️ Retry {attempt}/3 for batch {batch} ...")
            time.sleep(2 ** attempt)
        
        if not qna_data:
            print(f"❌ Skipping batch {batch} after 3 retries")
            continue
        
        # Filter duplicates
        new_items = []
        for item in qna_data:
            q = (item.get("question") or "").strip()
            if not q:
                continue
            q_norm = re.sub(r'\s+', ' ', q.lower())
            if q_norm not in seen:
                seen.add(q_norm)
                new_items.append(item)
        
        all_qna.extend(new_items)
        print(f"✅ Added {len(new_items)} new (Total: {len(all_qna)}/{num_questions})")
        
        if len(all_qna) >= num_questions:
            break
        
        time.sleep(1)  # Rate limiting
    
    if not all_qna:
        raise RuntimeError(f"No Q&A generated from {text_len} chars in {filepath}")
    
    all_qna = all_qna[:num_questions]
    
    # ✨ FORMAT OUTPUT
    formatted_output = []
    for i, q in enumerate(all_qna, start=1):
        question = q.get("question", "")
        answer = q.get("explanation", q.get("answer", ""))
        formatted_output.append(f"{i}. {question}\n{answer}\n")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_output))
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_qna, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Done! Saved for {base_name}:")
    print(f"📘 TXT → {output_txt}")
    print(f"📗 JSON → {output_json}")
    
    return {
        "success": True,
        "num_questions": len(all_qna),
        "input_chars": text_len,
        "output_txt": output_txt,
        "output_json": output_json
    }

# ============================================================
# 📁 FIXED Directory Processing
# ============================================================
def process_directory(directory: Path, output_dir: Path, num_questions: int = 300, batch_size: int = 25):
    """Process all TXT files in a directory."""
    txt_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".txt"):
                txt_files.append(Path(root) / fname)
    
    print(f"📚 Found {len(txt_files)} .txt files.\n")
    
    processed = 0
    failed = 0
    
    for fpath in txt_files:
        try:
            process_file(str(fpath), str(output_dir), num_questions, batch_size)
            processed += 1
            print(f"✅ {fpath.name}")
        except Exception as e:
            print(f"❌ {fpath.name}: {e}")
            failed += 1
    
    print(f"\n📊 SUMMARY: {processed} processed, {failed} failed")

# ============================================================
# 🚀 MAIN Handler (Production-grade)
# ============================================================
def handle_input_path(input_path: Path, output_dir: Path, num_questions: int = 300, batch_size: int = 25):
    """Handle TXT, ZIP, or directory input."""
    temp_dir = None
    
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == ".txt":
                process_file(str(input_path), str(output_dir), num_questions, batch_size)
            elif suffix == ".zip":
                print(f"📦 Extracting ZIP: {input_path}")
                zip_stem = input_path.stem
                temp_dir = Path(tempfile.mkdtemp(prefix="qna_zip_"))
                extract_dir = temp_dir / zip_stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                target_output = output_dir / zip_stem
                target_output.mkdir(parents=True, exist_ok=True)
                process_directory(extract_dir, target_output, num_questions, batch_size)
            else:
                raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .zip")
                
        elif input_path.is_dir():
            process_directory(input_path, output_dir, num_questions, batch_size)
        else:
            raise ValueError(f"Invalid input: {input_path}")
            
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================
# 🎛️ CLI (Enhanced)
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Q&A from text files (TXT/ZIP/Folder)")
    parser.add_argument("--input", required=True, help="Path to .txt/.zip file or folder")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-questions", type=int, default=300, help="Questions per file (default: 300)")
    parser.add_argument("--batch-size", type=int, default=25, help="Questions per API call (default: 25)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Q&A Generation Started!")
    handle_input_path(
        Path(args.input), 
        output_dir, 
        args.num_questions,
        args.batch_size
    )
    print("\n🎉 All Done!")
