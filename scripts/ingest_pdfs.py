"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     PDF INGESTION PIPELINE                               ║
║         Complete Pipeline: PDF → Text → Chunks → Embeddings              ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This is the DATA INGESTION SCRIPT for the Campus Companion RAG system.
It processes PDF documents and creates searchable embeddings in ChromaDB.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                  COMPLETE INGESTION FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [INPUT] PDF Files (data/pdfs/)                                     │
│      • academic_rules.pdf                                           │
│      • hostel_guidelines.pdf                                        │
│      • exam_regulations.pdf                                         │
│       ↓                                                             │
│  [STEP 1] Extract Text (PDFProcessor)                               │
│      • Try text extraction first (fast)                             │
│      • Fall back to OCR if needed (slow but accurate)               │
│      • Output: Plain text strings                                   │
│       ↓                                                             │
│  [STEP 2] Chunk Text (TextChunker)                                  │
│      • Split into 512-word chunks                                   │
│      • Add 50-word overlap                                          │
│      • Preserve metadata (filename, pages)                          │
│       ↓                                                             │
│  [STEP 3] Generate Embeddings (THIS FILE)                           │
│      • Use sentence-transformers (all-MiniLM-L6-v2)                 │
│      • Convert each chunk → 384-dim vector                          │
│      • ChromaDB handles this automatically!                         │
│       ↓                                                             │
│  [STEP 4] Store in ChromaDB (data/rag_docs/)                        │
│      • Persistent vector database                                   │
│      • Fast similarity search                                       │
│      • Ready for RAG queries!                                       │
│       ↓                                                             │
│  [OUTPUT] Searchable Knowledge Base                                 │
│      • Used by: core/rag.py                                         │
│      • Powers: "How to calculate CGPA?" queries                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT THIS SCRIPT DOES:
─────────────────────────────────────────────────────────────────────────
1. Scans data/pdfs/ directory for PDF files
2. Extracts text from each PDF (with OCR fallback)
3. Chunks text into 512-word segments with overlap
4. Generates embeddings using sentence-transformers
5. Stores in ChromaDB for semantic search
6. Displays statistics (documents processed, chunks created)

🚀 HOW TO RUN:
─────────────────────────────────────────────────────────────────────────
Method 1 - Run directly:
    python3 scripts/ingest_pdfs.py

Method 2 - From project root:
    python3 -m scripts.ingest_pdfs

What happens:
    ═══════════════════════════════════════════════════════════════
    Starting PDF Ingestion Pipeline
    ═══════════════════════════════════════════════════════════════
    
    [1/4] Extracting text from PDFs...
    ✓ academic_rules.pdf - 2,500 words (text extraction)
    ✓ hostel_guidelines.pdf - 1,800 words (OCR)
    ✓ exam_regulations.pdf - 3,200 words (text extraction)
    
    [2/4] Chunking text...
    ✓ Created 15 chunks (avg: 512 words/chunk)
    
    [3/4] Preparing documents for embedding...
    ✓ Ready for ChromaDB ingestion
    
    [4/4] Storing in ChromaDB...
    ✓ Successfully added 15 documents
    ✓ Collection now has 15 documents
    
    ═══════════════════════════════════════════════════════════════
    Ingestion Complete! ✅
    ═══════════════════════════════════════════════════════════════

📊 EXAMPLE TRANSFORMATION:
─────────────────────────────────────────────────────────────────────────
Input: academic_rules.pdf (10 pages, 5000 words)

After Processing:
  ├─ Chunk 1 (512 words)
  │    Text: "CGPA Calculation Rules: The cumulative..."
  │    Embedding: [0.234, -0.156, 0.891, ...] (384 dims)
  │    Metadata: {filename: 'academic_rules.pdf', pages: 10}
  │
  ├─ Chunk 2 (512 words)
  │    Text: "...grade point average is calculated..."
  │    Embedding: [0.445, 0.223, -0.334, ...] (384 dims)
  │    Metadata: {filename: 'academic_rules.pdf', pages: 10}
  │
  └─ ... (8 more chunks)

Stored in ChromaDB:
  • Fast similarity search
  • Automatically indexed
  • Query: "how is CGPA calculated?"
    → Returns: Chunks 1, 2 (highest similarity scores)

🔧 CONFIGURATION:
─────────────────────────────────────────────────────────────────────────
PDF_DIR = "data/pdfs/"
  • Where to find PDF files
  • Can be changed via constructor parameter

DB_PATH = "data/rag_docs/"
  • Where ChromaDB stores data
  • Persistent storage (survives restarts)

COLLECTION_NAME = "campus_docs"
  • Name of ChromaDB collection
  • Can have multiple collections for different purposes

CHUNK_SIZE = 512 words
  • How many words per chunk
  • Adjust in TextChunker initialization

CHUNK_OVERLAP = 50 words
  • Overlap between chunks
  • Prevents context loss

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
  • Sentence transformer model
  • 384 dimensions, fast, good quality
  • Downloaded automatically on first run (~90MB)

⚙️ COMPONENTS USED:
─────────────────────────────────────────────────────────────────────────
1. PDFProcessor (pdf_processor.py)
   • PyPDF2: Basic text extraction
   • pdfplumber: Better for tables
   • Tesseract OCR: For scanned PDFs
   
2. TextChunker (chunking.py)
   • Word-based sliding window
   • Preserves metadata
   
3. ChromaDB
   • Vector database
   • Handles embeddings automatically
   • No manual FAISS/Pinecone setup needed
   
4. sentence-transformers
   • all-MiniLM-L6-v2 model
   • Converts text → vectors
   • Managed by ChromaDB

💡 EMBEDDING EXPLAINED:
─────────────────────────────────────────────────────────────────────────
What is an embedding?
  • Numerical representation of text meaning
  • Each chunk → 384-number vector
  • Similar meanings → similar vectors

Example:
  "CGPA calculation rules"     → [0.23, -0.15, 0.89, ...]
  "how to calculate grades"    → [0.25, -0.14, 0.87, ...]  (similar!)
  "hostel food menu"           → [-0.45, 0.67, -0.12, ...] (different)

When user asks: "How do I calculate my CGPA?"
  1. Convert query → embedding [0.24, -0.16, 0.88, ...]
  2. Find chunks with similar embeddings (cosine similarity)
  3. Return top 3 most relevant chunks
  4. LLM generates answer from those chunks

🔄 RE-INGESTION:
─────────────────────────────────────────────────────────────────────────
To update documents:
  1. Add/modify PDFs in data/pdfs/
  2. Run: python3 scripts/ingest_pdfs.py
  3. ChromaDB will ADD new documents (won't delete old)
  
To start fresh:
  1. Delete: data/rag_docs/ folder
  2. Run: python3 scripts/ingest_pdfs.py
  3. Clean ChromaDB created from scratch

📝 IMPORTANT NOTES:
─────────────────────────────────────────────────────────────────────────
• First run downloads embedding model (~90MB) - be patient!
• OCR requires tesseract-ocr installed: brew install tesseract
• Large PDFs take time (1-2 mins for 100-page PDF with OCR)
• ChromaDB is persistent - data survives script restarts
• Safe to run multiple times (adds new docs, doesn't duplicate)

⚠️ TROUBLESHOOTING:
─────────────────────────────────────────────────────────────────────────
Error: "No such file or directory: data/pdfs"
  → Create folder: mkdir -p data/pdfs
  → Add some PDF files

Error: "tesseract not found"
  → Install: brew install tesseract (macOS)
  → Or disable OCR: PDFProcessor(ocr_enabled=False)

Error: "ChromaDB initialization failed"
  → Delete data/rag_docs/ and try again
  → Check permissions

Error: "Out of memory"
  → Process PDFs in smaller batches
  → Reduce chunk_size or process fewer files
"""


# ===========================================================================
# IMPORTS
# ===========================================================================



# ═══════════════════════════════════════════════════════════════════════
# ADD PROJECT ROOT TO PYTHON PATH
# ═══════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════
# STANDARD LIBRARIES
# ══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════════════
# PDF TO VECTOR DB INGESTION PIPELINE
# ═════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════




# ===========================================================================
# RUN SCRIPT
# ===========================================================================
