# =============================== DAY - 2 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          RAG (RETRIEVAL AUGMENTED GENERATION) SYSTEM                     ║
║        Semantic Search and Document Retrieval for Campus AI              ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This is the QUERY ENGINE of the Campus Companion RAG system.
It performs semantic search to find relevant document chunks for user questions.

This file is the FINAL STEP in the teaching sequence - where RAG actually happens!

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                   COMPLETE RAG ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [INGESTION PHASE] - Done Once                                      │
│  ═══════════════════════════════════════════════════════════════    │
│   1. PDF files (data/pdfs/)                                         │
│   2. Extract text (scripts/pdf_processor.py)                        │
│   3. Chunk text (scripts/chunking.py)                               │
│   4. Generate embeddings (core/embeddings.py)                       │
│   5. Store in ChromaDB (scripts/ingest_pdfs.py)                     │
│                                                                     │
│  [QUERY PHASE] - Every User Request (THIS FILE!)                    │
│  ═══════════════════════════════════════════════════════════════    │
│   1. User asks: "How to calculate CGPA?"                            │
│   2. Convert question → embedding (384-dim vector)                  │
│   3. Search ChromaDB for similar embeddings                         │
│   4. Retrieve top-k most relevant chunks                            │
│   5. Return chunks to LLM                                           │
│   6. LLM generates answer from retrieved context                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT IS RAG?
─────────────────────────────────────────────────────────────────────────
RAG = Retrieval Augmented Generation

Problem: LLMs don't know your specific information
  • Campus rules change every semester
  • Hostel policies are unique to your college
  • LLM training data doesn't include this

Traditional Solutions (Don't Work Well):
  ❌ Fine-tuning: Expensive, slow, needs retraining for updates
  ❌ Prompt stuffing: Hit context limits with large docs
  ❌ Manual updates: Time-consuming, error-prone

RAG Solution (Best Approach):
  ✅ Store documents as searchable embeddings
  ✅ Find relevant chunks on-demand
  ✅ Give only relevant context to LLM
  ✅ Update by re-ingesting documents (no retraining!)

💡 RAG QUERY PROCESS (STEP BY STEP):
─────────────────────────────────────────────────────────────────────────
Let's trace a real query: "How to calculate CGPA?"

STEP 1: User Query
  Input: "How to calculate CGPA?"

STEP 2: Query Embedding
  • Convert query to 384-dim vector using same model as ingestion
  • Query vector: [0.234, -0.156, 0.891, ...]
  • Model: all-MiniLM-L6-v2 (same as embedding phase)

STEP 3: Similarity Search in ChromaDB
  • Compare query vector with all stored chunk vectors
  • Calculate cosine similarity: similarity = dot(query, chunk) / (|query| * |chunk|)
  • Similarity ranges from 0 (unrelated) to 1 (identical meaning)

  Example results:
    Chunk 1: "CGPA calculation involves..." → similarity: 0.89 ⭐⭐⭐
    Chunk 2: "Grade points are computed..." → similarity: 0.76 ⭐⭐
    Chunk 15: "Hostel food menu includes..." → similarity: 0.12 ❌

STEP 4: Retrieve Top-K Chunks
  • Sort by similarity score
  • Take top 3 chunks (configurable)
  • Filter by minimum score (default: 0.3)

STEP 5: Format Results
  Return:
  [
    {'content': 'CGPA calculation...', 'relevance_score': 0.89},
    {'content': 'Grade points...', 'relevance_score': 0.76},
    {'content': 'Final CGPA...', 'relevance_score': 0.68}
  ]

STEP 6: LLM Answer Generation (core/response.py)
  • Combine retrieved chunks into context
  • Send to LLM: "Based on: [chunks], answer: [query]"
  • LLM generates accurate, context-aware response

🔍 SEMANTIC SEARCH EXPLAINED:
─────────────────────────────────────────────────────────────────────────
Semantic = Understanding Meaning, Not Just Keywords

Traditional Keyword Search:
  Query: "CGPA calculation"
  Matches: Only documents with exact words "CGPA" and "calculation"
  Misses: "How to compute grade point average"

Semantic Search (Embedding-Based):
  Query: "CGPA calculation" → [0.23, -0.15, 0.89, ...]
  Finds similar meanings:
    ✓ "CGPA calculation" (exact match)
    ✓ "How to compute grade point average" (same meaning!)
    ✓ "GPA computation rules" (related concept)
    ✓ "Academic performance metrics" (broader topic)

This is why embeddings are powerful!

📊 EXAMPLE QUERY FLOW:
─────────────────────────────────────────────────────────────────────────
User: "Can I change my hostel?"

1. Query Embedding:
   "Can I change my hostel?" → [0.45, 0.23, -0.67, ...]

2. ChromaDB Search (behind the scenes):
   • Compares with 150 stored chunks
   • Finds top 3 matches:
     
     Rank 1 (score: 0.87):
       "Hostel changes are allowed after first semester.
        Students must submit request form to warden..."
     
     Rank 2 (score: 0.72):
       "Room swapping procedure: Fill form at hostel office.
        Approval takes 2-3 weeks..."
     
     Rank 3 (score: 0.65):
       "Hostel allocation policy: First-years assigned randomly.
        Second-years can request preferred hostel..."

3. Return to LLM:
   LLM receives these 3 chunks and generates:
   "Yes, you can change your hostel after the first semester.
    You need to submit a request form to the warden..."

💻 USAGE:
─────────────────────────────────────────────────────────────────────────
    from core.rag import RAGSystem, get_rag_system
    
    # Initialize RAG system (connects to ChromaDB)
    rag = RAGSystem()
    
    # Search for relevant documents
    results = rag.search_documents(
        query="How to calculate CGPA?",
        top_k=3,           # Return top 3 chunks
        min_score=0.3      # Minimum relevance threshold
    )
    
    # Use results
    for doc in results:
        print(f"Score: {doc['relevance_score']:.2f}")
        print(f"Content: {doc['content'][:100]}...")
    
    # Or use singleton instance (recommended for API)
    rag = get_rag_system()  # Reuses same instance
    results = rag.search_documents("hostel rules")

🔧 CONFIGURATION:
─────────────────────────────────────────────────────────────────────────
db_path = "data/rag_docs"
  • Where ChromaDB stores data
  • Persistent storage survives restarts

collection_name = "campus_docs"
  • Name of document collection
  • Can have multiple collections for different purposes

top_k = 3
  • Number of chunks to retrieve
  • 3-5 is usually optimal
  • Too few: might miss context
  • Too many: adds noise, uses more tokens

min_score = 0.3
  • Minimum relevance threshold (0-1)
  • 0.3 is permissive (broader results)
  • 0.6 is strict (only high-quality matches)
  • Adjust based on your needs

embedding_model = "all-MiniLM-L6-v2"
  • MUST match ingestion model!
  • 384 dimensions
  • Fast and accurate

⚡ PERFORMANCE:
─────────────────────────────────────────────────────────────────────────
Query Speed:
  • Small collection (100 chunks): ~10-50ms
  • Medium collection (1000 chunks): ~50-200ms
  • Large collection (10,000 chunks): ~200-500ms
  • ChromaDB uses HNSW index for fast search

Memory Usage:
  • Embeddings stay in ChromaDB (not in RAM)
  • Only loads what's needed for query
  • Efficient for large document sets

Scalability:
  • ChromaDB handles millions of documents
  • For production: consider managed ChromaDB or Pinecone
  • Can implement caching for frequent queries

🎓 WHY RAG OVER ALTERNATIVES?
─────────────────────────────────────────────────────────────────────────
1. vs Fine-Tuning:
   RAG: Update documents anytime, instant effect
   Fine-Tuning: Retrain entire model, expensive, slow

2. vs Prompt Stuffing:
   RAG: Only retrieve relevant chunks (efficient)
   Prompt Stuffing: Send entire docs, hit context limits

3. vs Vector Similarity Search Only:
   RAG: Combines retrieval + generation
   Vector Search: Only finds docs, doesn't answer

4. vs Knowledge Graphs:
   RAG: Simpler to implement and maintain
   Knowledge Graphs: Complex setup, rigid structure

📝 IMPORTANT NOTES:
─────────────────────────────────────────────────────────────────────────
• Query embedding MUST use same model as ingestion
• ChromaDB handles embedding automatically (no manual work!)
• Singleton pattern avoids re-initializing on every query
• Distance → Similarity conversion: similarity = 1 - distance
• Lower distance = higher similarity
• ChromaDB returns distances, we convert to similarity scores

⚠️ TROUBLESHOOTING:
─────────────────────────────────────────────────────────────────────────
Error: "Collection not found"
  → Run ingestion first: python scripts/ingest_pdfs.py
  → Check db_path matches ingestion path

Error: "No results returned"
  → Lower min_score threshold
  → Check if documents were ingested properly
  → Verify query is in English (model limitation)

Poor Results:
  → Increase top_k (more chunks)
  → Adjust min_score
  → Improve document chunking (smaller/larger chunks)
  → Try different embedding model

Slow Queries:
  → Check collection size (too many docs?)
  → Consider indexing options
  → Use singleton pattern (get_rag_system())
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════
# RAG SYSTEM CLASS
# ══════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════
# LEGACY QUERY FUNCTION FOR BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════



# ===========================================================================
# SINGLETON RAG SYSTEM INSTANCE
# ===========================================================================




# ===========================================================================
# GET RAG SYSTEM SINGLETON
# ===========================================================================
