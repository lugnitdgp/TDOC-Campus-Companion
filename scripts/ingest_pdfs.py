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
import os
import sys
from pathlib import Path
import logging
from typing import List,Dict


# ═══════════════════════════════════════════════════════════════════════
# ADD PROJECT ROOT TO PYTHON PATH
# ═══════════════════════════════════════════════════════════════════════

project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))



# ══════════════════════════════════════════════════════════════════════
# STANDARD LIBRARIES
# ══════════════════════════════════════════════════════════════════════

from scripts.pdf_processor import PDFProcessor
from scripts.chunking import TextChunker
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions




# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# PDF TO VECTOR DB INGESTION PIPELINE
# ═════════════════════════════════════════════════════════════════════

class PDFIngestionPipeline:
    
    def __init__(
        self,
        pdf_dir: str = "data/pdfs",
        db_path: str = "data/rag_docs",
        collection_name:str = "campus_docs",
        chunk_size:int = 512,
        chunk_overlap:int =50
    ):
        self.pdf_dir = Path(pdf_dir)
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        self.pdf_processor = PDFProcessor(ocr_enabled=True)
        self.chunker = TextChunker(chunk_size,chunk_overlap)

        try:
            
            self.client = chromadb.PersistentClient(
                path = str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

            self.collection =self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=sentence_transformer_ef,
                metadata={'description':"campus documents for rag"}
                )
            
            logger.info(f"chroma db is initialized {self.db_path}")

        except Exception as e:
            logger.error(f"failed to initialize the chromadb : {e}")
            return RuntimeError(f"chromadb initialization failed:{e}")
        
    def run(self):
        logger.info("="*60)
        logger.info("Starting PDF Ingestion Pipeline")
        logger.info("="*60)

        logger.info("\n[1/4] Extracting text from PFDS ...")
        pdf_results = self.pdf_processor.process_directory(self.pdf_dir)

        if not pdf_results: 
            logger.error("No PDFs processed. Exiting.")
            return
        
        logger.info(f"Processed {len(pdf_results)} PDFs")

        logger.info("\n[2/4] Chunking text ...")
        all_chunks=[]
        for pdf_data in pdf_results:
            metadata = {
                'filename': pdf_data['filename'],
                'pages': pdf_data['pages'],
                'method':pdf_data['method']
            }
            chunks = self.chunker.chunk_text(pdf_data['text'],metadata)
            all_chunks.extend(chunks)
        logger.info("Created {len(all_chunks)} chunks")

        logger.info("\n[3/4] Preparing documents for embedding...")
        documents=[]
        metadata=[]
        ids=[]

        for i,chunk in enumerate(all_chunks):
            documents.append(chunk['text'])
            metadata.append(chunk['metadata'])
            ids.append(f"chunk_{i}")

        logger.info(f"Prepared {len(documents)} documents")

        logger.info("\n[4/4] Storing in vector database")

        try:
            existing_ids = self.collection.get()['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except:
            pass
        
        batch_size=1000
        for i in range(0,len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta= metadata[i:i+batch_size]
            batch_ids= ids[i:i+batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            logger.info(f"Added batch {i//batch_size+1} ({len(batch_docs)})")

        logger.info(f"Stored {len(documents)} documents with embeddings in ChromaDB")


        logger.info("\n"+"="*60)
        logger.info("ingestion complete")
        logger.info("="*60)
        logger.info(f"PDFS processed: {len(pdf_results)}")
        logger.info(f"Chunks created: {len(all_chunks)}")
        logger.info(f"Documents stored: {len(documents)}")
        logger.info(f"Collection: {len(self.collection_name)}")
        logger.info(f"Database: {self.db_path}")

        self._test_retrieval()


    def _test_retrieval(self):
        logger.info("\n [TEST] Running sample query")

        results= self.collection.query(
            query_texts= ["CGPA calculation"],
            n_results=3
        )

        if results['documents']:
            logger.info("Retrieval working! Sample results:")
            for i,doc in enumerate(results['documents'][0][:2],1):
                logger.info(f" {i}. {doc[:100]}...")





      



            
# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def main():
    try:
        PDF_DIR="data/pdfs"
        DB_PATH= "data/rag_docs"


        Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
        Path(DB_PATH).mkdir(parents=True, exist_ok=True)

        pdf_files=list(Path(PDF_DIR).glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {PDF_DIR}")
            logger.info(f"Please add PDF Files to {PDF_DIR}/ directory")
            logger.info(f"Example: cp your document.pdf {PDF_DIR}/")
            raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
        
        logger.info(f"Found {len(pdf_files)} PDF file(s)")

        pipeline=PDFIngestionPipeline(
            pdf_dir=PDF_DIR,
            db_path=DB_PATH,
            chunk_size=512,
            chunk_overlap=50
        )
        



# ===========================================================================
# RUN SCRIPT
# ===========================================================================
        pipeline.run()

    except Exception as e:
        logger.error(f"pipeline failed: {e}")
        raise
    
if __name__ == "__main__":
    main()