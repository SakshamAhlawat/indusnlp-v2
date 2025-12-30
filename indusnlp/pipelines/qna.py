"""
Q&A Generation Pipeline - Generate educational Q&A pairs using Google Gemini.
Refactored from test_qna_generation.py
"""

import os
import re
import json
import time
import math
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

class QnAPipeline:
    """
    Q&A Generation Pipeline using Google Gemini API.
    Generates educational question-answer pairs from Hindi/bilingual text.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Q&A pipeline.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not configured. Set it in environment variables.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    def _clean_json_response(self, raw: str) -> Optional[List[Dict]]:
        """Clean and parse JSON from Gemini response."""
        if not raw:
            return None

        # ✅ FIXED: Proper regex pattern
        raw = re.sub(r"``````", "", raw.strip())
        print(f"🔍 Raw after cleaning: {repr(raw[:200])}...")  # Debug
        
        start, end = raw.find("["), raw.rfind("]") + 1
        if start == -1 or end <= 0:
            print(f"❌ No JSON array found in: {repr(raw[:100])}")
            return None

        text = raw[start:end]
        
        try:
            result = json.loads(text)
            if not isinstance(result, list):
                print(f"❌ Expected list, got: {type(result)}")
                return None
            print(f"✅ Parsed {len(result)} Q&A items")
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Failed text: {repr(text[:100])}")
            return None
    
    def _generate_batch(self, text: str, seen: set[str]) -> Optional[List[Dict]]:
        """Generate a single batch of Q&A pairs."""
        import google.generativeai as genai
        
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
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.5,
                    top_p=0.95,
                    top_k=40,
                ),
            )
            return self._clean_json_response(response.text)
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            return None
    
    def generate_qna(
        self,
        text: str,
        num_questions: int = 25,
        batch_size: int = 25,
        chunk_size: int = 6000,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Generate Q&A pairs from text.
        
        Args:
            text: Source text for Q&A generation
            num_questions: Total number of questions to generate
            batch_size: Questions per API call
            chunk_size: Text chunk size for each batch
            max_retries: Max retries per batch on failure
            
        Returns:
            List of Q&A dictionaries
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")
        
        all_qna: List[Dict] = []
        seen: set[str] = set()
        num_batches = math.ceil(num_questions / batch_size)
        
        for batch in range(1, num_batches + 1):
            # Ensure chunk_size doesn't exceed text length
            effective_chunk_size = min(chunk_size, len(text))
            start = ((batch - 1) * effective_chunk_size) % len(text)
            chunk_text = text[start:start + effective_chunk_size]
            
            # Skip if chunk is too small
            if len(chunk_text.strip()) < 100:
                print(f"⚠️ Batch {batch}: Chunk too small ({len(chunk_text.strip())} chars), skipping.")
                continue
            
            qna_data = None
            for attempt in range(1, max_retries + 1):
                qna_data = self._generate_batch(chunk_text, seen)
                if qna_data:
                    break
                time.sleep(2)
            
            if not qna_data:
                print(f"⚠️ Batch {batch}: Failed after {max_retries} retries.")
                continue
            
            # Filter duplicates - ✅ ADDED TYPE CHECK
            for item in qna_data:
                if not isinstance(item, dict):  # ✅ Safety check
                    print(f"⚠️ Skipping non-dict item: {type(item)}")
                    continue
                    
                q = item.get("question", "").strip()
                if not q:
                    continue
                q_norm = re.sub(r'\s+', ' ', q.lower())
                if q_norm not in seen:
                    seen.add(q_norm)
                    all_qna.append(item)
            
            if len(all_qna) >= num_questions:
                break
            
            time.sleep(1)
        
        if not all_qna:
            raise RuntimeError("No Q&A could be generated from the provided text. Check text quality and API key.")
        
        return all_qna[:num_questions]
    
    def process_file(
        self,
        filepath: str,
        output_dir: str,
        num_questions: int = 300,
        batch_size: int = 25
    ) -> Dict[str, Any]:
        """
        Process a single text file and generate Q&A.
        
        Args:
            filepath: Path to input text file
            output_dir: Directory for output files
            num_questions: Number of questions to generate
            batch_size: Questions per batch
            
        Returns:
            Dict with results including generated Q&A
        """
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        output_txt = os.path.join(output_dir, f"{base_name}_QA.txt")
        output_json = os.path.join(output_dir, f"{base_name}_QA.json")
        
        # Generate Q&A
        all_qna = self.generate_qna(text, num_questions, batch_size)
        
        # Format and save TXT
        formatted_output = []
        for i, q in enumerate(all_qna, start=1):
            question = q.get("question", "")
            answer = q.get("explanation", q.get("answer", ""))
            formatted_output.append(f"{i}. {question}\n{answer}\n")
        
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(formatted_output))
        
        # Save JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_qna, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "num_questions": len(all_qna),
            "output_txt": output_txt,
            "output_json": output_json,
            "qna": all_qna
        }
    
    def process_input(
        self,
        input_path: Path,
        output_dir: Path,
        num_questions: int = 300,
        batch_size: int = 25
    ) -> Dict[str, Any]:
        """
        Handle different input types: TXT file, ZIP file, or directory.
        
        Args:
            input_path: Path to input
            output_dir: Output directory
            num_questions: Questions per file
            batch_size: Questions per batch
            
        Returns:
            Dict with processing results
        """
        results = {"processed": 0, "failed": 0, "files": [], "qna": None}
        temp_dir = None
        
        try:
            if input_path.is_file():
                suffix = input_path.suffix.lower()
                if suffix == ".txt":
                    # Single text file
                    result = self.process_file(
                        str(input_path),
                        str(output_dir),
                        num_questions,
                        batch_size
                    )
                    results["processed"] = 1
                    results["files"].append(input_path.name)
                    results["qna"] = result["qna"]
                    
                elif suffix == ".zip":
                    # ZIP file
                    zip_stem = input_path.stem
                    temp_dir = Path(tempfile.mkdtemp(prefix="qna_zip_"))
                    extract_dir = temp_dir / zip_stem
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    
                    with zipfile.ZipFile(input_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    target_output = output_dir / zip_stem
                    target_output.mkdir(parents=True, exist_ok=True)
                    results = self._process_directory(extract_dir, target_output, num_questions, batch_size)
                    
            elif input_path.is_dir():
                results = self._process_directory(input_path, output_dir, num_questions, batch_size)
                
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return results
    
    def _process_directory(
        self,
        directory: Path,
        output_dir: Path,
        num_questions: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Process all TXT files in a directory."""
        results = {"processed": 0, "failed": 0, "files": []}
        
        txt_files = []
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith(".txt"):
                    txt_files.append(Path(root) / fname)
        
        for fpath in txt_files:
            try:
                self.process_file(str(fpath), str(output_dir), num_questions, batch_size)
                results["processed"] += 1
                results["files"].append(fpath.name)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                results["failed"] += 1
        
        return results


# Module-level convenience function
def generate_qna(text: str, num_questions: int = 25, api_key: Optional[str] = None) -> List[Dict]:
    """
    Generate Q&A pairs from text.
    
    Args:
        text: Source text
        num_questions: Number of questions
        api_key: Optional Gemini API key
        
    Returns:
        List of Q&A dictionaries
    """
    pipeline = QnAPipeline(api_key=api_key)
    return pipeline.generate_qna(text, num_questions)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Q&A from text files.")
    parser.add_argument("--input", required=True, help="Path to .txt/.zip file or folder.")
    parser.add_argument("--output", required=True, help="Directory to store generated Q&A files.")
    parser.add_argument("--num-questions", type=int, default=300, help="Number of questions per file")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = QnAPipeline()
    results = pipeline.process_input(Path(args.input), output_dir, args.num_questions)
    
    print(f"\n🎉 Q&A Generation Complete!")
    print(f"✅ Processed: {results['processed']}, Failed: {results['failed']}")
