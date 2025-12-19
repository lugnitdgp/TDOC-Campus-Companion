# ================================ DAY - 4 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                   RESPONSE GENERATION SYSTEM                             ║
║         Final Answer Generation for Campus Companion Chatbot             ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This is the FINAL STEP in the query pipeline - generating the actual response
that users see. It takes retrieved data and converts it into natural language.

This is the "voice" of the chatbot.

🔗 HOW IT FITS IN THE COMPLETE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] USER QUERY                                                     │
│      "How to calculate CGPA?"                                       │
│       ↓                                                             │
│  [2] INTENT CLASSIFICATION (core/classifier.py)                     │
│      Intent: "rag" (0.90 confidence)                                │
│       ↓                                                             │
│  [3] DATA RETRIEVAL                                                 │
│      RAG Search (core/rag.py) → Retrieved 3 relevant chunks         │
│       ↓                                                             │
│  [4] RESPONSE GENERATION (THIS FILE!) ← YOU ARE HERE                │
│      Chunks + Query → Natural Language Answer                       │
│       ↓                                                             │
│  [5] USER RECEIVES ANSWER                                           │
│      "CGPA is calculated by dividing sum of grade points..."        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────
Takes RAW DATA and converts it to USER-FRIENDLY ANSWERS:

INPUT (Raw Data):
  • RAG chunks: ["CGPA is calculated...", "Grade points are...", ...]
  • Database results: {phone: "+91-xxx", name: "Roy canteen"}
  • Intent: "rag" or "db_contact" or "db_location"

OUTPUT (Natural Language):
  • Formatted answer: "CGPA is calculated by dividing the sum..."
  • Sources: [academic_rules.pdf, student_handbook.pdf]
  • Confidence: 0.85

🔄 COMPLETE RESPONSE FLOW:
─────────────────────────────────────────────────────────────────────────

Example Query: "How to calculate CGPA?"

STEP 1: Intent Classification
  → Intent: "rag" (policy question)
  → Route to: RAG system

STEP 2: Query Refinement (Optional)
  Original: "How to calculate CGPA?"
  Refined: "CGPA calculation method steps"
  Why? Better semantic search results

STEP 3: RAG Search
  → Search ChromaDB for: "CGPA calculation method steps"
  → Retrieved chunks:
    [1] "CGPA is calculated by dividing sum of grade points..." (0.89)
    [2] "Grade points for each course are computed..." (0.76)
    [3] "Final CGPA appears on semester report..." (0.68)

STEP 4: Context Building
  → Combine chunks (max 2000 chars)
  → Add source labels: [Source 1], [Source 2], etc.
  → Result: Single context string for LLM

STEP 5: LLM Answer Generation
  → Send to Mistral-7B:
    System: "Answer only from context"
    Context: [Combined chunks]
    Query: "How to calculate CGPA?"
  
  → LLM Response:
    "CGPA is calculated by dividing the sum of grade points
     by total credits. For each course, grade points are
     computed by multiplying grade value by course credits..."

STEP 6: Format & Return
  {
    "answer": "CGPA is calculated by...",
    "sources": [
      {"filename": "academic_rules.pdf", "relevance": 0.89},
      {"filename": "student_handbook.pdf", "relevance": 0.76}
    ],
    "confidence": 0.78,
    "method": "rag_hf_llm"
  }

📊 RESPONSE METHODS:
─────────────────────────────────────────────────────────────────────────

1. RAG RESPONSE (Policy Questions)
   ─────────────────────────────────
   Query: "How to calculate CGPA?"
   
   Process:
   1. Refine query for better search
   2. Search ChromaDB (semantic similarity)
   3. Retrieve top-k relevant chunks
   4. Build context from chunks
   5. Send to LLM for natural answer
   6. Format with sources
   
   Output:
   • Natural language explanation
   • Source documents listed
   • High confidence (0.7-0.9)

2. DATABASE CONTACT RESPONSE
   ──────────────────────────
   Query: "Roy canteen phone number"
   
   Process:
   1. Query contacts database
   2. Fetch: name, phone, email
   3. Format with emojis
   
   Output:
   🍽️ Roy Canteen
   📞 Phone: +91-xxx-xxxx
   📧 Email: roy@campus.edu

3. DATABASE LOCATION RESPONSE
   ───────────────────────────
   Query: "Where is room AB101?"
   
   Process:
   1. Query locations database
   2. Fetch: room, building, floor
   3. Format with emojis
   
   Output:
   🚪 Room AB101
   🏢 Building: Academic Block
   🏗️ Floor: 1st Floor

4. AI FALLBACK RESPONSE
   ─────────────────────
   Query: "What's the weather?"
   
   Process:
   1. No database/RAG match
   2. Return capability message
   
   Output:
   "I can help with academics, contacts, and campus locations.
    Please ask something related to campus information."

🤖 LLM INTEGRATION (Mistral-7B via HuggingFace):
─────────────────────────────────────────────────────────────────────────

WHY USE LLM?
  Raw chunks: Hard to read, fragmented
  LLM output: Natural, coherent, conversational

BEFORE LLM (Raw chunks):
  "CGPA is calculated by dividing sum grade points total credits
   Grade points computed multiplying grade value course credits
   Final CGPA appears semester report card transcript"

AFTER LLM (Natural language):
  "CGPA is calculated by dividing the sum of grade points by your
   total credits. For each course, grade points are computed by
   multiplying the grade value by the course credits. Your final
   CGPA will appear on your semester report card and transcript."

LLM CONFIGURATION:
  • Model: mistralai/Mistral-7B-Instruct-v0.2
  • API: HuggingFace Inference API (free tier)
  • Max tokens: 512 (enough for detailed answers)
  • Temperature: 0.3 (factual, not creative)
  • Timeout: 120 seconds

FALLBACK STRATEGY:
  If LLM unavailable:
  ✓ Still works! Returns formatted chunks
  ✗ Less natural but still useful
  ✓ No dependency on external API

⚡ PERFORMANCE CHARACTERISTICS:
─────────────────────────────────────────────────────────────────────────

Response Generation Times:
  • Database queries: ~10-50ms (instant)
  • RAG search: ~50-200ms (fast)
  • LLM generation: ~1-3 seconds (slow but acceptable)
  • Total: ~1.5-3.5 seconds for RAG queries

Optimization:
  • Singleton pattern (reuse LLM connection)
  • Context length limit (2000 chars)
  • Top-k retrieval (5 docs max)
  • Query refinement cache (future)

Token Usage (HuggingFace Free Tier):
  • Context: ~500 tokens
  • Response: ~200 tokens
  • Total: ~700 tokens per query
  • Free tier: 1000 requests/day

🔧 KEY FEATURES:
─────────────────────────────────────────────────────────────────────────

1. QUERY REFINEMENT
   Improves search results by rephrasing queries
   Example: "cgpa rule?" → "CGPA calculation rules and requirements"

2. CONTEXT BUILDING
   Combines multiple chunks into coherent context
   Limits length to avoid token overflow
   Labels sources: [Source 1], [Source 2]

3. CONFIDENCE SCORING
   Averages relevance scores from top-3 chunks
   Range: 0.0 (no confidence) to 1.0 (very confident)
   Helps users trust responses

4. SOURCE TRACKING
   Lists which documents contributed to answer
   Enables verification and transparency
   Shows relevance score per source

5. GRACEFUL FALLBACK
   Works without LLM (returns formatted chunks)
   Handles API failures silently
   Always returns something useful

💻 USAGE:
─────────────────────────────────────────────────────────────────────────

Simple (Auto-detect intent):
    from core.response import generate_response
    
    result = generate_response("How to calculate CGPA?")
    print(result['answer'])
    print(f"Confidence: {result['confidence']}")

RAG-specific (Policy questions):
    from core.response import generate_rag_response
    
    result = generate_rag_response("CGPA calculation rules")
    print(result['answer'])
    for source in result['sources']:
        print(f"- {source['filename']} ({source['relevance']})")

With explicit intent:
    result = generate_response("Roy canteen phone", intent="db_contact")
    print(result['answer'])

Using class directly:
    from core.response import ResponseGenerator
    
    gen = ResponseGenerator()
    result = gen.generate_response("hostel rules")

📝 RESPONSE FORMAT:
─────────────────────────────────────────────────────────────────────────

All response methods return a dictionary:

{
    "answer": str,           # Main response text
    "sources": List[Dict],   # Source documents
    "confidence": float,     # 0.0-1.0 confidence score
    "method": str           # How answer was generated
}

Methods:
  • "rag_hf_llm": RAG + Mistral-7B (best)
  • "rag_basic": RAG without LLM (fallback)
  • "rag_no_results": RAG found nothing
  • "db_contact": Database contact lookup
  • "db_location": Database location lookup
  • "ai_fallback": Out-of-scope response

⚠️ IMPORTANT NOTES:
─────────────────────────────────────────────────────────────────────────
• Requires HUGGINGFACEHUB_ACCESS_TOKEN in .env for LLM
• Works without LLM but responses less natural
• Context limited to 2000 chars (prevents token overflow)
• Query refinement only for queries >3 words (efficiency)
• Singleton pattern avoids reinitializing LLM
• All methods return dict format for consistency
"""

# =======================================================================
# IMPORTS
# =======================================================================





# =======================================================================
# LOAD ENV VARIABLES
# =======================================================================




# =======================================================================
# LOGGING SETUP
# =======================================================================





# =======================================================================
# RESPONSE GENERATOR CLASS
# =======================================================================



# ----------------------------------------------------------------------
# GLOBAL HELPERS
# ----------------------------------------------------------------------

