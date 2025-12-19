# ================================ DAY 5 & 6 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         CHAT API ENDPOINT                                ║
║        The Central Brain of the Campus Companion System                  ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This file is the MAIN ENTRY POINT for all user questions.
Whenever a user asks something, the request ALWAYS reaches this file.

Think of this file as:
🧠 The brain + 🧭 traffic controller + 🗣️ mouth of the system

It does NOT store data.
It does NOT train AI.
It ONLY coordinates everything.

🔗 HOW THIS FILE FITS IN THE SYSTEM:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE SYSTEM FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] USER TYPES: "Roy canteen phone"                                │
│       ↓                                                             │
│  [2] FRONTEND (frontend.py)                                         │
│      • Sends POST to /api/chat                                      │
│       ↓                                                             │
│  [3] BACKEND (api/main.py)                                          │
│      • Routes to THIS FILE                                          │
│       ↓                                                             │
│  [4] THIS FILE (api/routers/chat.py) ← YOU ARE HERE!                │
│      ┌─────────────────────────────────────────────────┐            │
│      │ STEP 1: CLASSIFY INTENT                         │            │
│      │   Uses: core/classifier.py                      │            │
│      │   Result: "db_contact" (85% confidence)         │            │
│      └─────────────────────────────────────────────────┘            │
│       ↓                                                             │
│      ┌─────────────────────────────────────────────────┐            │
│      │ STEP 2: ROUTE TO HANDLER                        │            │
│      │   Calls: try_get_contact(text, session)        │             │
│      │   Searches: Canteen, Faculty, Warden tables    │             │
│      │   Result: "Roy Canteen: 9876543210"            │             │
│      └─────────────────────────────────────────────────┘            │
│       ↓                                                             │
│      ┌─────────────────────────────────────────────────┐            │
│      │ STEP 3: FORMAT RESPONSE                         │            │
│      │   Uses: core/response.py (Mistral-7B AI)       │             │
│      │   Result: Natural language response            │             │
│      └─────────────────────────────────────────────────┘            │
│       ↓                                                             │
│      Returns JSON: {                                                │
│        "answer": "Roy Canteen's phone number is...",                │
│        "intent": "db_contact",                                      │
│        "confidence": 0.85                                           │
│      }                                                              │
│       ↓                                                             │
│  [5] FRONTEND DISPLAYS RESPONSE                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


🎯 RESPONSIBILITIES OF THIS FILE:
─────────────────────────────────────────────────────────────────────────
1. POST /api/chat endpoint - receives user queries
2. Validates request (ensures text field exists)
3. Classifies intent using core/classifier.py
4. Routes to appropriate handler:
   • db_contact → try_get_contact() → Search Canteen/Faculty/Warden
   • db_location → try_get_location() → Search Room/Building
   • faculty_info → try_get_faculty() → Search Faculty table
   • rag → try_get_rag() → Search ChromaDB documents
   • small_talk → handle_small_talk() → Friendly response
   • ai_fallback → fallback_ai_response() → Mistral-7B AI
5. Formats response using core/response.py
6. Returns JSON to frontend

📊 DATA FLOW EXAMPLES:
─────────────────────────────────────────────────────────────────────────
Example 1: Contact Query
  User: "Roy canteen phone"
  ↓
  Classify: db_contact (90%)
  ↓
  Handler: try_get_contact()
    → Searches: Canteen table WHERE name LIKE '%roy%'
    → Finds: Roy Canteen, Phone: 9876543210
  ↓
  Format: "Roy Canteen's contact number is 9876543210..."
  ↓
  Return: {"answer": "...", "intent": "db_contact", "confidence": 0.9}

Example 2: Document Query (RAG)
  User: "How to calculate CGPA?"
  ↓
  Classify: rag (85%)
  ↓
  Handler: try_get_rag()
    → Searches: ChromaDB embeddings (semantic search)
    → Finds: 3 relevant chunks from academic_rules.pdf
  ↓
  Format: AI reads chunks and generates answer
  ↓
  Return: {"answer": "CGPA is calculated by...", "intent": "rag"}

Example 3: No Data Found (Fallback)
  User: "What's the weather?"
  ↓
  Classify: ai_fallback (70%)
  ↓
  Handler: fallback_ai_response()
    → Returns: Campus-focused guidance message
  ↓
  Return: {"answer": "I'm Campus Companion...", "used_fallback": true}


🔑 KEY COMPONENTS:
─────────────────────────────────────────────────────────────────────────
1. ChatRequest/ChatResponse - Pydantic models for validation
2. chat() - Main endpoint function
3. Handler Functions:
   • try_get_contact() - Search contact databases
   • try_get_location() - Search location databases
   • try_get_faculty() - Search faculty database
   • try_get_rag() - Search RAG documents (semantic)
   • handle_small_talk() - Friendly greetings
4. Integration Points:
   • core/classifier.py - Intent classification
   • core/response.py - Response formatting
   • core/fallback_message.py - AI fallback
   • core/rag.py - Document search
   • db/models.py - Database tables

💡 HANDLER LOGIC EXPLAINED:
─────────────────────────────────────────────────────────────────────────
Each handler follows this pattern:

def try_get_X(text: str, session) -> Optional[str]:
    '''
    Search for X in database
    Returns: Raw data string if found, None if not found
    '''
    1. Extract keywords from query
    2. Search database with fuzzy matching (ILIKE)
    3. Validate results (check if entity name matches)
    4. Format as string
    5. Return data OR None

This decouples data retrieval from response formatting!

🚨 ERROR HANDLING:
─────────────────────────────────────────────────────────────────────────
• Empty query → Friendly prompt
• Classification error → AI fallback
• Database error → Error message + log
• Formatting error → Return raw data
• All errors logged with traceback

📝 IMPORTANT NOTES:
─────────────────────────────────────────────────────────────────────────
• Always close database session (finally block)
• Extensive debug logging for troubleshooting
• Response always includes: answer, intent, confidence
• RAG results truncated to prevent huge responses (500 chars/chunk)
• Fallback always provides helpful response (never "I don't know")
"""

# ═════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════







# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================







# ============================================================================
# ROUTER SETUP
# ============================================================================







# ============================================================================
# HANDLER FUNCTIONS (keep all your existing handlers)
# ============================================================================







