"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     TEXT CHUNKING FOR RAG                                ║
║              Splits Documents into Searchable Chunks                     ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This file splits long documents into smaller CHUNKS for RAG (Retrieval Augmented Generation).
It's a critical preprocessing step before creating embeddings.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] PDF FILES (data/pdfs/)                                         │
│      • academic_rules.pdf (50 pages, 20,000 words)                  │
│      • hostel_guidelines.pdf (30 pages, 12,000 words)               │
│       ↓                                                             │
│  [2] PDF PROCESSOR (scripts/pdf_processor.py)                       │
│      • Extracts text from PDFs                                      │
│      • Output: Full text strings                                    │
│       ↓                                                             │
│  [3] THIS FILE (scripts/chunking.py) ← YOU ARE HERE!                │
│      • Splits text into 512-word chunks                             │
│      • Creates 50-word overlap between chunks                       │
│      • Output: List of chunks with metadata                         │
│       ↓                                                             │
│  [4] EMBEDDING GENERATION (scripts/ingest_pdfs.py)                  │
│      • Converts each chunk to 384-dim vector                        │
│       ↓                                                             │
│  [5] CHROMADB STORAGE (data/rag_docs/)                              │
│      • Stores chunks + embeddings                                   │
│      • Ready for semantic search!                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHY CHUNKING IS NECESSARY:
─────────────────────────────────────────────────────────────────────────
Problem: LLMs have context limits
  • Mistral-7B: ~8000 tokens (~6000 words)
  • Can't process 50-page PDF all at once
  
Solution: Break into smaller chunks
  • Each chunk is self-contained
  • Overlap ensures context isn't lost
  • Search returns only relevant chunks

Example:
  Original: 20,000-word PDF
  After chunking: 40 chunks × 512 words each
  Query: "CGPA calculation"
  Search returns: 3 most relevant chunks (1536 words total)
  LLM processes: Only those 3 chunks, not entire PDF!

💡 CHUNKING STRATEGY:
─────────────────────────────────────────────────────────────────────────
Word-Based Sliding Window:
  
  CHUNK 1: Words 0-512
  ┌─────────────────────────────────────────┐
  │ CGPA is calculated by summing all...    │
  │ ...grade points divided by credits...   │
  │ ...following rules apply: (1) minimum   │
  └─────────────────────────────────────────┘
                    ↓ (overlap 50 words)
  CHUNK 2: Words 462-974
  ┌─────────────────────────────────────────┐
  │ ...following rules apply: (1) minimum   │
  │ ...passing grade is D, (2) grades are   │
  │ ...recorded per semester, (3) final...  │
  └─────────────────────────────────────────┘

Benefits of Overlap:
  • Ensures important sentences aren't cut off
  • Maintains context across chunks
  • If query matches boundary, both chunks retrieved

🔧 CONFIGURATION:
─────────────────────────────────────────────────────────────────────────
chunk_size = 512 words
  • ~384-512 English words
  • ~2500-3000 characters
  • Sweet spot for semantic search
  
chunk_overlap = 50 words
  • ~10% overlap
  • Balances context vs redundancy
  • Prevents important info loss at boundaries

📊 CHUNKING EXAMPLE:
─────────────────────────────────────────────────────────────────────────
Input Text (1000 words):
  "CGPA Calculation Rules: The CGPA is calculated by dividing the sum
   of grade points by total credits. Grade points for each course are
   computed by multiplying the grade value by the course credits..."

Output Chunks:
  Chunk 1 (words 0-512):
    text: "CGPA Calculation Rules: The CGPA is..."
    metadata: {filename: 'academic_rules.pdf', page: 1}
    length: 512 words

  Chunk 2 (words 462-974):  # Overlaps by 50 words
    text: "...grade points by total credits. Grade points..."
    metadata: {filename: 'academic_rules.pdf', page: 1-2}
    length: 512 words

  Chunk 3 (words 924-1000):  # Last chunk (shorter)
    text: "...computed by multiplying the grade value..."
    metadata: {filename: 'academic_rules.pdf', page: 2}
    length: 76 words

💻 USAGE:
─────────────────────────────────────────────────────────────────────────
    from scripts.chunking import TextChunker
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    
    # Chunk a document
    text = "Long document text here..."
    metadata = {'filename': 'rules.pdf', 'page': 1}
    chunks = chunker.chunk_text(text, metadata)
    
    # Result:
    # [
    #   {'text': '...', 'metadata': {...}, 'length': 512},
    #   {'text': '...', 'metadata': {...}, 'length': 512},
    #   ...
    # ]

⚙️ HOW TO ADJUST:
─────────────────────────────────────────────────────────────────────────
Larger Chunks (More Context):
    chunker = TextChunker(chunk_size=1024, chunk_overlap=100)
    ↑ Better for complex queries needing more context
    ↓ But: More tokens used, slower search

Smaller Chunks (More Precise):
    chunker = TextChunker(chunk_size=256, chunk_overlap=25)
    ↑ Better for specific factual queries
    ↓ But: May lose context, more chunks to search

📝 NOTES:
─────────────────────────────────────────────────────────────────────────
• Chunks are measured in WORDS, not characters
• Text is cleaned (extra whitespace removed)
• Metadata preserved for traceability
• If text < chunk_size, returns single chunk
• Overlap prevents infinite loop (capped at chunk_size-1)
"""

# ===========================================================================
# IMPORTS
# ===========================================================================



# ===========================================================================
# TEXT CHUNKER CLASS
# ===========================================================================
