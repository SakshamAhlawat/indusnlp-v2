# ============================================================
# 📂 Upload TXT File(s), Normal Folder, or ZIP Folder → Generate 300 Q&A
# ============================================================
import argparse
import os, json, re, time, math, zipfile, tempfile, shutil
from pathlib import Path
import google.generativeai as genai

# 🔑 Configure Gemini
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDaZkLuaucHSHK6QXqav94itSoNCrRBZM8")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ============================================================
# 🧹 JSON Cleaning
# ============================================================
def clean_json(raw):
    if not raw:
        return None
    raw = re.sub(r"```json|```", "", raw.strip())
    start, end = raw.find("["), raw.rfind("]") + 1
    if start == -1 or end <= 0:
        return None
    text = raw[start:end]
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        return json.loads(text)
    except:
        return None

# ============================================================
# 🤖 Generate One Batch of Detailed Q&A
# ============================================================
def generate_detailed_qna(text, seen):
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
                temperature=0.5,   # 🔥 REDUCED
                top_p=0.95,
                top_k=40,
            ),
        )
        return clean_json(response.text)
    except Exception as e:
        print("⚠️ Gemini error:", e)
        return None

# ============================================================
# 🚀 Generate 300 Q&A for a Single File
# ============================================================
def process_file(filepath, output_dir):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    TOTAL_Q = 300
    BATCH_SIZE = 25
    CHUNK_SIZE = 6000
    MAX_RETRIES = 3

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_txt = os.path.join(output_dir, f"{base_name}_300_QA.txt")
    output_json = os.path.join(output_dir, f"{base_name}_300_QA.json")

    all_qna = []
    seen = set()
    num_batches = math.ceil(TOTAL_Q / BATCH_SIZE)

    print(f"\n🧠 Generating {TOTAL_Q} questions for: {base_name}\n")

    for batch in range(1, num_batches + 1):
        print(f"⚙️ Batch {batch}/{num_batches} ...")

        start = ((batch - 1) * CHUNK_SIZE) % len(text)
        chunk_text = text[start:start + CHUNK_SIZE]

        qna_data = None
        for attempt in range(1, MAX_RETRIES + 1):
            qna_data = generate_detailed_qna(chunk_text, seen)
            if qna_data:
                break
            print(f"⚠️ Retry {attempt}/{MAX_RETRIES} for batch {batch} ...")
            time.sleep(5)

        if not qna_data:
            print(f"❌ Skipping batch {batch} after {MAX_RETRIES} attempts.")
            continue

        # Filter duplicates
        new_items = []
        for item in qna_data:
            q = item.get("question", "").strip()
            if not q:
                continue
            q_norm = re.sub(r'\s+', ' ', q.lower())
            if q_norm not in seen:
                seen.add(q_norm)
                new_items.append(item)

        all_qna.extend(new_items)
        print(f"✅ Added {len(new_items)} new (Total: {len(all_qna)}/{TOTAL_Q})")

        if len(all_qna) >= TOTAL_Q:
            break

        time.sleep(3)

    all_qna = all_qna[:TOTAL_Q]

    # ============================================================
    # ✨ FORMAT OUTPUT TO TXT (Numbered Q&A)
    # ============================================================
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
    print(f"📘 TXT  → {output_txt}")
    print(f"📗 JSON → {output_json}\n")


# ============================================================
def handle_input_path(input_path: Path, output_dir: Path):
    temp_dir = None

    try:
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == ".txt":
                process_file(str(input_path), str(output_dir))
            elif suffix == ".zip":
                zip_stem = input_path.stem
                temp_dir = Path(tempfile.mkdtemp(prefix="qna_zip_"))
                extract_dir = temp_dir / zip_stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                target_output = output_dir / zip_stem
                target_output.mkdir(parents=True, exist_ok=True)
                _process_directory(extract_dir, target_output)
            else:
                print(f"⚠️ Unsupported file type: {input_path}")
        elif input_path.is_dir():
            _process_directory(input_path, output_dir)
        else:
            print(f"⚠️ Input path not found: {input_path}")
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _process_directory(directory: Path, output_dir: Path):
    txt_files = []
    for root, _, files_in_dir in os.walk(directory):
        for fname in files_in_dir:
            if fname.lower().endswith(".txt"):
                txt_files.append(Path(root) / fname)

    print(f"📚 Found {len(txt_files)} .txt files.\n")

    for fpath in txt_files:
        process_file(str(fpath), str(output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Q&A from text files.")
    parser.add_argument("--input", required=True, help="Path to .txt/.zip file or folder.")
    parser.add_argument("--output", required=True, help="Directory to store generated Q&A files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    handle_input_path(Path(args.input), output_dir)