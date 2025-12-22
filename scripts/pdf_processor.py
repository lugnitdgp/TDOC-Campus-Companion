# ============================== DAY - 2 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                  PDF TEXT EXTRACTION WITH OCR                            ║
║            Extracts Text from PDFs with Smart Fallback                   ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This file extracts text from PDF documents using multiple strategies:
1. Direct text extraction (fast) - for digital PDFs
2. OCR (slow but accurate) - for scanned/image-based PDFs

It's the FIRST STEP in the RAG data ingestion pipeline.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] PDF FILES (data/pdfs/) ← INPUT                                 │
│      • academic_rules.pdf                                           │
│      • hostel_guidelines.pdf                                        │
│       ↓                                                             │
│  [2] THIS FILE (scripts/pdf_processor.py) ← YOU ARE HERE!           │
│      • Try PyPDF2 first (fast text extraction)                      │
│      • Fallback to OCR if needed (slower but works on scans)        │
│      • Output: Plain text strings                                   │
│       ↓                                                             │
│  [3] TEXT CHUNKING (scripts/chunking.py)                            │
│      • Split text into 512-word chunks                              │
│       ↓                                                             │
│  [4] EMBEDDINGS (scripts/ingest_pdfs.py)                            │
│      • Convert chunks to vectors                                    │
│       ↓                                                             │
│  [5] CHROMADB STORAGE (data/rag_docs/)                              │
│      • Store for semantic search                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHY MULTIPLE EXTRACTION METHODS?
─────────────────────────────────────────────────────────────────────────
Different PDFs require different approaches:

1. DIGITAL PDFs (text-based):
   • Created from Word, LaTeX, web browsers
   • Text is embedded in PDF structure
   • Fast extraction: PyPDF2, pdfplumber
   • Example: Modern academic papers, official documents

2. SCANNED PDFs (image-based):
   • Created from photocopies, phone scans
   • Text is pixels, not characters
   • Requires OCR (Optical Character Recognition)
   • Example: Old books, handwritten forms, photos

SMART FALLBACK STRATEGY:
  1. Try PyPDF2 first (< 1 second)
  2. Check if result has enough text (>100 chars)
  3. If insufficient, use OCR (~10-30 seconds per page)

💡 EXTRACTION METHODS COMPARED:
─────────────────────────────────────────────────────────────────────────
Method         Speed      Quality    Works On
─────────────────────────────────────────────────────────────────────────
PyPDF2         ⚡ Fast    Good       Digital PDFs only
pdfplumber     ⚡ Fast    Better     Digital PDFs, tables
Tesseract OCR  🐌 Slow    Excellent  Everything (images, scans)

📊 EXAMPLE OUTPUT:
─────────────────────────────────────────────────────────────────────────
Input: academic_rules.pdf

Output:
{
    'filename': 'academic_rules.pdf',
    'text': 'CGPA Calculation Rules\\n\\nThe CGPA is calculated...',
    'pages': 10,
    'method': 'text_extraction',  # or 'ocr'
    'path': '/full/path/to/academic_rules.pdf'
}

💻 USAGE:
─────────────────────────────────────────────────────────────────────────
    from scripts.pdf_processor import PDFProcessor
    
    # Initialize processor
    processor = PDFProcessor(ocr_enabled=True)
    
    # Process single PDF
    result = processor.extract_text_from_pdf('data/pdfs/rules.pdf')
    print(result['text'])
    
    # Process entire directory
    results = processor.process_directory('data/pdfs')
    for doc in results:
        print(f"{doc['filename']}: {len(doc['text'])} chars")

🔧 DEPENDENCIES:
─────────────────────────────────────────────────────────────────────────
Required packages:
  • PyPDF2: Basic PDF text extraction
  • pdfplumber: Better extraction for complex layouts
  • pytesseract: Python wrapper for Tesseract OCR
  • pdf2image: Convert PDF pages to images for OCR
  • Pillow (PIL): Image processing

System requirements for OCR:
  macOS:   brew install tesseract poppler
  Ubuntu:  apt install tesseract-ocr poppler-utils
  Windows: Download from GitHub (tesseract, poppler)

⚠️ TROUBLESHOOTING:
─────────────────────────────────────────────────────────────────────────
Error: "tesseract not found"
  → Install: brew install tesseract (macOS)
  → Or disable OCR: PDFProcessor(ocr_enabled=False)

Error: "poppler not found"
  → Install: brew install poppler (macOS)
  → Needed for pdf2image

Error: "Extraction returned empty string"
  → Enable OCR: PDFProcessor(ocr_enabled=True)
  → Check if PDF is corrupted

Slow OCR performance:
  → Reduce DPI: convert_from_path(path, dpi=150)
  → Default is 300 DPI (high quality, slower)
  → 150 DPI: 2-4x faster, slightly lower accuracy

📝 NOTES:
─────────────────────────────────────────────────────────────────────────
• First extraction attempt uses PyPDF2 (fastest)
• OCR only triggered if text < 100 characters
• OCR processes at 300 DPI for best quality
• Each page takes ~10-15 seconds with OCR
• 100-page PDF with OCR: ~15-25 minutes
• Consider batch processing large PDFs overnight
"""


# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════

import os
import logging
from pathlib import Path
from typing import List,Dict

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# PDF PROCESSOR CLASS
# ===========================================================================

class PDFProcessor:
    
    def __init__(self,ocr_enabled:bool = True):
        self.ocr_enabled = ocr_enabled

    def extract_text_from_pdf(self,pdf_path:str)->Dict[str,any]:
        
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found:{pdf_path}")
        
        logger.info(f"Processing:{pdf_path.name}")

        text=self._extract_text_pypdf(pdf_path)
        method="text_extraction"

        if not text or len(text.strip())<100:
            
            if self.ocr_enabled:
                logger.info(f"Text extraction insufficient.Using OCR for {pdf_path.name}")
                text = self._extract_text_ocr(pdf_path)
                method = "ocr"
            else:
                logger.warning(f"OCR disabled.Skipping {pdf_path.name}")

        with open(pdf_path,'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)

        return{
            'filename':pdf_path.name,
            'text':text.strip(),
            'pages':page_count,
            'method':method,
            'path':str(pdf_path)
        }  

    def _extract_text_pypdf(self,pdf_path:Path)->str:

        try :
            with open(pdf_path,'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""

                for page in pdf_reader.pages:
                    text+= page.extract_text()+"\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed : {e}")
            return ""
    
    def _extract_text_ocr(self,pdf_path: Path) -> str:
        try:
            images = convert_from_path(pdf_path,dpi=300)
            text= ""

            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")

                page_text = pytesseract.image_to_string(image,lang='eng')

                text+= page_text+'\n'
            return text
        
        except Exception as e:
            logger.error(f"OCR extraction failed:{e}") 
            return ""

    def process_directory(self,pdf_dir:str)-> List[Dict]:
        pdf_dir=Path(pdf_dir)

        pdf_files = list(pdf_dir.glob('*.pdf'))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")

        results = []

        for pdf_file in pdf_files:
            
            try:
                result = self.extract_text_from_pdf(pdf_file)
                results.append(result)
            
            except Exception as e:
                logger.error(f"failed to process {pdf_file.name}:{e}")

        return results
