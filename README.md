# IndusNLP v2.0

Hindi/Indic NLP toolkit with OCR, text cleaning, and Q&A generation capabilities.

## Features

- **OCR Processing**: Extract text from PDF files using Mistral OCR
- **Text Cleaning**: Advanced Hindi text cleaning with transliteration support (IndicTrans2)
- **Q&A Generation**: Generate educational Q&A pairs using Google Gemini

## Quick Start

### Prerequisites

- **Python 3.9 or 3.10** (required for IndicTrans2/fairseq compatibility)
  - ⚠️ Python 3.11+ is NOT supported due to fairseq dataclass incompatibility
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation with uv (Recommended)

```bash
# Install uv if not already installed
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync

# Activate virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Alternative: Installation with pip

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
MISTRAL_API_KEY=your_mistral_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Running the Server

```bash
# Development
python -m indusnlp.app

# Production (with gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 indusnlp.app:app
```

## Git & .gitignore

This repository includes a `.gitignore` tuned for this project. It intentionally ignores:

- **Python build and cache files**: `__pycache__/`, `*.py[cod]`, `build/`, `dist/`, `*.egg-info`, etc.
- **Virtual environments**: `.venv/`, `venv/`, `ENV/`, `env/`
- **uv metadata**: `.uv/`, `uv.lock`
- **IDE and OS junk**: `.idea/`, `.vscode/`, `.DS_Store`, `Thumbs.db`, swap files (`*.swp`, `*.swo`, `*~`)
- **Environment files**: `.env`, `.env.local`
- **Project-specific generated outputs**:
  - Log and error files: `*.log`, `errors.txt`
  - Cleaning outputs: `*_cleaned.txt`
  - Q&A outputs: `*_QA.txt`, `*_QA.json`

Only source code, config (`pyproject.toml`, `.env.example`) and docs (`README.md`) are meant to be committed. All temporary outputs from OCR, cleaning, and Q&A generation stay untracked.

## API Endpoints

### Health Check

```
GET /
GET /health
```

Returns service status and available endpoints.

---

### OCR Endpoint

```
POST /api/ocr
```

Extract text from PDF files using Mistral OCR.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - PDF or ZIP file containing PDFs

**Response (single PDF):**
```json
{
  "success": true,
  "filename": "document.pdf",
  "text": "Extracted text content..."
}
```

**Response (ZIP):** Returns a ZIP file containing `.txt` files

**Example:**
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/ocr
```

---

### Text Cleaning Endpoint

```
POST /api/clean
```

Clean Hindi/Indic text with optional transliteration.

**Request Options:**

1. **JSON Body:**
```json
{
  "text": "Your text to clean...",
  "transliterate": false,
  "filter_badwords": true,
  "filter_punctuation": false
}
```

2. **File Upload:**
- Content-Type: `multipart/form-data`
- Body: `file` - TXT or ZIP file

**Query Parameters:**
- `transliterate` (bool, default: false) - Enable English to Hindi transliteration
- `filter_badwords` (bool, default: true) - Filter profanity
- `filter_punctuation` (bool, default: false) - Keep only punctuated lines

**Response:**
```json
{
  "success": true,
  "original_length": 1500,
  "cleaned_length": 1200,
  "text": "Cleaned text content..."
}
```

**Example:**
```bash
# JSON request
curl -X POST http://localhost:5000/api/clean \
  -H "Content-Type: application/json" \
  -d '{"text": "Your Hindi text here...", "transliterate": true}'

# File upload
curl -X POST -F "file=@document.txt" http://localhost:5000/api/clean?transliterate=true
```

---

### Q&A Generation Endpoint

```
POST /api/qna
```

Generate educational Q&A pairs from text using Google Gemini.

**Request Options:**

1. **JSON Body:**
```json
{
  "text": "Your educational text...",
  "num_questions": 25,
  "batch_size": 25
}
```

2. **File Upload:**
- Content-Type: `multipart/form-data`
- Body: `file` - TXT or ZIP file

**Query Parameters:**
- `num_questions` (int, default: 25) - Number of questions to generate
- `batch_size` (int, default: 25) - Questions per API call

**Response:**
```json
{
  "success": true,
  "num_questions": 25,
  "qna": [
    {
      "type": "MCQ",
      "question": "Question text...",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "Explanation..."
    },
    {
      "type": "Summarization",
      "question": "Question text...",
      "answer": "Answer text..."
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/qna \
  -H "Content-Type: application/json" \
  -d '{"text": "Educational content...", "num_questions": 10}'
```

---

## Project Structure

```
indusnlp-v2/
├── indusnlp/
│   ├── __init__.py
│   └── app.py              # Flask API application
├── filters/
│   ├── __init__.py
│   ├── HindiTextCleaner.py # Hindi text cleaning with IndicTrans2
│   ├── textcleaner.py      # General text cleaning utilities
│   ├── badwords_en_hi_hiR.py
│   └── data/               # JSON data files
├── pyproject.toml          # Project configuration (uv/pip)
├── .env.example            # Environment variables template
└── README.md
```

## Dependencies

Key dependencies (managed via `pyproject.toml`):

- **Flask** - Web framework
- **ai4bharat-transliteration** - IndicTrans2 for transliteration
- **indic-nlp-library** - Indic NLP resources
- **mistralai** - Mistral OCR API client
- **google-generativeai** - Google Gemini API client
- **gunicorn** - Production WSGI server
- **lxml** - HTML/XML processing

## Migration from v1

### IndicTrans v1 → v2 Changes

The old `indictrans` library has been replaced with `ai4bharat-transliteration`:

**Before (v1):**
```python
from indictrans import Transliterator
transliterator = Transliterator(source='eng', target='hin')
result = transliterator.transform("hello")
```

**After (v2):**
```python
from ai4bharat.transliteration import XlitEngine
transliterator = XlitEngine("hi", beam_width=4, rescore=True)
result = transliterator.translit_word("hello", topk=1)["hi"][0]
```

## License

MIT License
