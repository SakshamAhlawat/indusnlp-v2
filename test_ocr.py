# ============================================================
# 📄 OCR Processing: PDF Files and ZIP Archives
# ============================================================
import argparse
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = "fCx1qVhbannccU1UNZMnISojbnE3iZeA"


client = Mistral(api_key=api_key)


# ============================================================
# Helper: get all PDF file paths recursively
# ============================================================
def get_all_pdfs(folder):
    pdf_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
    return pdf_files


# ============================================================
# Process a single PDF file
# ============================================================
def process_pdf(pdf_path: str, output_dir: Path, base_dir: Path = None):
    """
    Process a single PDF file using Mistral OCR API.
    
    Args:
        pdf_path: Full path to the PDF file
        output_dir: Directory where output .txt files will be saved
        base_dir: Base directory to compute relative paths (for preserving structure)
    """
    try:
        print(f"📄 Processing: {pdf_path}")

        # Compute output .txt path mirroring directory structure
        if base_dir:
            relative_path = os.path.relpath(pdf_path, base_dir)
            txt_path = output_dir / (os.path.splitext(relative_path)[0] + ".txt")
        else:
            # If no base_dir, just use the filename
            pdf_name = Path(pdf_path).stem
            txt_path = output_dir / f"{pdf_name}.txt"
        
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Upload file to Mistral
        with open(pdf_path, "rb") as f:
            uploaded_file = client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": f
                },
                purpose="ocr"
            )

        # Get signed URL
        file_url = client.files.get_signed_url(file_id=uploaded_file.id)

        # Run OCR
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": file_url.url
            },
            include_image_base64=False,
        )

        # Write OCR output (Markdown/text)
        with open(txt_path, "w", encoding="utf-8") as f:
            for page_num, page in enumerate(response.pages, start=1):
                page_text = getattr(page, "markdown", "") or getattr(page, "text", "")
                f.write(f"\n{page_text.strip()}\n")

        print(f"✅ Saved: {txt_path}\n")
        return True

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        error_path = output_dir / "errors.txt"
        with open(error_path, "a", encoding="utf-8") as ef:
            ef.write(f"[ERROR processing {pdf_path}: {e}]\n")
        return False


# ============================================================
# Process directory containing PDFs
# ============================================================
def process_directory(directory: Path, output_dir: Path):
    """Process all PDF files in a directory recursively."""
    pdf_files = get_all_pdfs(str(directory))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{directory}'. Exiting.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in '{directory}'\n")
    
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"📄 [{idx}/{len(pdf_files)}] Processing: {pdf_path}")
        process_pdf(pdf_path, output_dir, base_dir=directory)


# ============================================================
# Handle input path (file, directory, or ZIP)
# ============================================================
def handle_input_path(input_path: Path, output_dir: Path):
    """Handle different input types: PDF file, ZIP file, or directory."""
    temp_dir = None

    try:
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == ".pdf":
                # Single PDF file
                output_dir.mkdir(parents=True, exist_ok=True)
                process_pdf(str(input_path), output_dir)
            elif suffix == ".zip":
                # ZIP file - extract and process PDFs
                zip_stem = input_path.stem
                temp_dir = Path(tempfile.mkdtemp(prefix="ocr_zip_"))
                extract_dir = temp_dir / zip_stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"📦 Extracting ZIP: {input_path}")
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                target_output = output_dir / zip_stem
                target_output.mkdir(parents=True, exist_ok=True)
                process_directory(extract_dir, target_output)
            else:
                print(f"⚠️ Unsupported file type: {input_path}. Expected .pdf or .zip")
        elif input_path.is_dir():
            # Directory containing PDFs
            output_dir.mkdir(parents=True, exist_ok=True)
            process_directory(input_path, output_dir)
        else:
            print(f"⚠️ Input path not found: {input_path}")
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Main entry point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="OCR processing for PDF files and ZIP archives.")
    parser.add_argument("--input", required=True, help="Path to .pdf/.zip file or folder containing PDFs.")
    parser.add_argument("--output", required=True, help="Directory to store OCR output .txt files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        exit(1)
    
    handle_input_path(input_path, output_dir)
    print(f"\n🎉 All OCR text files saved under '{output_dir}/'")
