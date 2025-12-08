"""IndusNLP - Hindi/Indic NLP toolkit."""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.textcleaner import TextCleaner
from filters.HindiTextCleaner import HindiTextCleaner
from filters.badwords_en_hi_hiR import badword_list

from indusnlp.pipelines.cleaning import CleaningPipeline, master_cleaning_pipeline
from indusnlp.pipelines.ocr import OCRPipeline
from indusnlp.pipelines.qna import QnAPipeline

__version__ = "2.0.0"
__all__ = [
    "TextCleaner",
    "HindiTextCleaner", 
    "badword_list",
    "CleaningPipeline",
    "master_cleaning_pipeline",
    "OCRPipeline",
    "QnAPipeline",
]
