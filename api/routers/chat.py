# ================================ DAY 6 ================================ #
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

from typing import Optional 
from fastapi import APIRouter
from pydantic import BaseModel
from core.classifier import classify_detailed
from core.response import format_response
from core.fallback_message import fallback_ai_response
from db.session import SessionLocal
from db import models
from sqlmodel import select
from core.rag import query_rag

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    text: str 
    
    class Config:
        # Example shown in /docs page
        schema_extra = {
            "example": {
                "text": "Roy canteen phone number"
            }
        }


class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    used_fallback: bool
    is_multi_intent: bool = False
    all_intents: list = []


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):  # ← CHANGED: Now uses Pydantic model
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Extract Query Text (now from Pydantic model)
    # ═══════════════════════════════════════════════════════════════════
    text = request.text.strip()  # ← CHANGED: Access via request.text
    print(f"[DEBUG] Received chat request: {text}")
    
    # Validation: Check if empty after stripping whitespace
    if not text:
        return ChatResponse(
            answer="Please ask me something! Try: 'Roy canteen phone' or 'CGPA rules'",
            intent="empty_query",
            confidence=1.0,
            used_fallback=False
        )
    
    print(f"\n{'='*70}")
    print(f"[CHAT] Received query: {text}")
    print(f"{'='*70}")
    
    # ═══════════════════════════════════════════════════════════════════
    # DATABASE SESSION
    # ═══════════════════════════════════════════════════════════════════
    session = SessionLocal()
    
    try:
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: CLASSIFY INTENT
        # ═══════════════════════════════════════════════════════════════
        print(f"[STEP 1] Classifying intent...")
        classification = classify_detailed(text)
        
        print(f"[RESULT] Intent: {classification.primary_intent}")
        print(f"[RESULT] Confidence: {classification.confidence:.3f}")
        print(f"[RESULT] Needs Fallback: {classification.needs_fallback}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: ROUTE TO HANDLER
        # ═══════════════════════════════════════════════════════════════
        raw_data = None
        intent = classification.primary_intent
        
        print(f"\n[STEP 2] Routing to handler for intent: {intent}")
        
        # Contact queries
        if intent == 'db_contact':
            print(f"[HANDLER] Searching contact databases...")
            raw_data = try_get_contact(text, session)
            if raw_data:
                print(f"[SUCCESS] Found contact data ({len(raw_data)} chars)")
            else:
                print(f"[INFO] No contact data found")
        
        # Location queries
        elif intent == 'db_location':
            print(f"[HANDLER] Searching location databases...")
            raw_data = try_get_location(text, session)
            if raw_data:
                print(f"[SUCCESS] Found location data ({len(raw_data)} chars)")
            else:
                print(f"[INFO] No location data found")
        
        # Faculty queries
        elif intent == 'faculty_info':
            print(f"[HANDLER] Searching faculty database...")
            raw_data = try_get_faculty(text, session)
            if raw_data:
                print(f"[SUCCESS] Found faculty data ({len(raw_data)} chars)")
            else:
                print(f"[INFO] No faculty data found")
        
        # Document queries (RAG)
        elif intent == 'rag':
            print(f"[HANDLER] Searching document embeddings (RAG)...")
            raw_data = try_get_rag(text)
            if raw_data:
                print(f"[SUCCESS] Found RAG data ({len(raw_data)} chars)")
            else:
                print(f"[INFO] No relevant documents found")
        
        # Greetings
        elif intent == 'small_talk':
            print(f"[HANDLER] Handling small talk...")
            return handle_small_talk(text)
        
        # AI Fallback
        elif intent == 'ai_fallback':
            print(f"[HANDLER] Using AI fallback (Mistral-7B)...")
            ai_answer = fallback_ai_response(text)
            print(f"[SUCCESS] AI generated response ({len(ai_answer)} chars)")
            
            return ChatResponse(
                answer=ai_answer,
                intent=intent,
                confidence=classification.confidence,
                used_fallback=True,
                is_multi_intent=classification.is_multi_intent,
                all_intents=[
                    {
                        "name": i.intent,
                        "confidence": i.confidence,
                        "method": i.method
                    }
                    for i in classification.all_intents[:3]
                ]
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: FORMAT RESPONSE
        # ═══════════════════════════════════════════════════════════════
        
        if raw_data:
            # Data found - format it
            print(f"\n[STEP 3] Formatting response with AI...")
            
            try:
                formatted_answer = format_response(text, raw_data)
                print(f"[SUCCESS] Response formatted ({len(formatted_answer)} chars)")
                
                return ChatResponse(
                    answer=formatted_answer,
                    intent=intent,
                    confidence=classification.confidence,
                    used_fallback=False,
                    is_multi_intent=classification.is_multi_intent,
                    all_intents=[
                        {
                            "name": i.intent,
                            "confidence": i.confidence,
                            "method": i.method
                        }
                        for i in classification.all_intents[:3]
                    ]
                )
            
            except Exception as format_error:
                # If formatting fails, return raw data
                print(f"[WARNING] Formatting failed: {format_error}")
                print(f"[FALLBACK] Returning raw data")
                
                return ChatResponse(
                    answer=raw_data,
                    intent=intent,
                    confidence=classification.confidence,
                    used_fallback=False,
                    is_multi_intent=classification.is_multi_intent
                )
        
        else:
            # No data found - use AI fallback
            print(f"\n[STEP 3] No data found, using AI fallback...")
            
            ai_answer = fallback_ai_response(text)
            print(f"[SUCCESS] AI generated fallback ({len(ai_answer)} chars)")
            
            return ChatResponse(
                answer=ai_answer,
                intent=intent,
                confidence=classification.confidence,
                used_fallback=True,
                is_multi_intent=classification.is_multi_intent,
                all_intents=[
                    {
                        "name": i.intent,
                        "confidence": i.confidence,
                        "method": i.method
                    }
                    for i in classification.all_intents[:3]
                ]
            )
    
    except Exception as e:
        # Error handling
        print(f"\n[ERROR] Exception in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            answer=(
                "I encountered an error processing your request. "
                "Please try again or contact campus administration."
            ),
            intent="error",
            confidence=0.0,
            used_fallback=True
        )
    
    finally:
        session.close()
        print(f"{'='*70}\n")


# ============================================================================
# HANDLER FUNCTIONS (keep all your existing handlers)
# ============================================================================

def try_get_contact(text: str, session) -> Optional[str]:
    text_lower = text.lower()
    
    # Extract potential entity names (words that might be canteen names)
    words = text_lower.split()
    
    # Common noise words to ignore
    noise_words = ['give', 'me', 'the', 'phone', 'number', 'contact', 
                   'of', 'for', 'canteen', 'cafeteria', 'mess', 'no', 'tell']
    
    # Get potential entity names (non-noise words)
    potential_names = [w for w in words if w not in noise_words and len(w) > 2]
    
    print(f"[DEBUG] Potential entity names: {potential_names}")
    
    # ═══════════════════════════════════════════════════════════════
    # SEARCH CANTEEN
    # ═══════════════════════════════════════════════════════════════
    
    if any(keyword in text_lower for keyword in ['canteen', 'cafeteria', 'mess']):
        print(f"[DEBUG] Searching canteens...")
        
        # Get all canteens
        canteens = session.query(models.Canteen).all()
        
        if not canteens:
            print(f"[DEBUG] No canteens in database")
        else:
            print(f"[DEBUG] Found {len(canteens)} canteens")
            
            matched_canteen = None
            best_match_count = 0
            
            for canteen in canteens:
                canteen_name_lower = canteen.name.lower()
                
                # Count how many potential names match
                match_count = 0
                for potential_name in potential_names:
                    if potential_name in canteen_name_lower:
                        match_count += 1
                        print(f"[DEBUG] '{potential_name}' found in '{canteen.name}'")
                
                # Keep track of best match
                if match_count > best_match_count:
                    best_match_count = match_count
                    matched_canteen = canteen
            
            # If specific name match found, return it
            if matched_canteen and best_match_count > 0:
                print(f"[DEBUG] Best match: {matched_canteen.name} ({best_match_count} matches)")
                
                return (
                    f"🍽️ **{matched_canteen.name}** (Canteen)\n"
                    f"📞 Phone: {matched_canteen.phone}\n"
                    f"📧 Email: {matched_canteen.email}\n"
                    f"📍 Location: {matched_canteen.location or 'N/A'}"
                )
            
            # If no specific match but user just asked for "canteen"
            elif len(potential_names) == 0 and canteens:
                # Generic query like "canteen phone" - return first one
                matched_canteen = canteens[0]
                print(f"[DEBUG] Generic canteen query, returning: {matched_canteen.name}")
                
                return (
                    f"🍽️ **{matched_canteen.name}** (Canteen)\n"
                    f"📞 Phone: {matched_canteen.phone}\n"
                    f"📧 Email: {matched_canteen.email}\n"
                    f"📍 Location: {matched_canteen.location or 'N/A'}"
                )
            
            else:
                # User specified a canteen name but no match found
                print(f"[DEBUG] No canteen found matching: {potential_names}")
                # Don't return here - continue to check faculty/warden
    
    # ═══════════════════════════════════════════════════════════════
    # SEARCH FACULTY - IMPROVED!
    # ═══════════════════════════════════════════════════════════════
    
    # NEW: Search faculty if:
    # 1. Explicit keywords present OR
    # 2. We have potential names (could be faculty name)
    
    should_search_faculty = (
        any(keyword in text_lower for keyword in ['faculty', 'professor', 'dr', 'teacher']) or
        len(potential_names) >= 2  # Has at least first + last name
    )
    
    if should_search_faculty:
        print(f"[DEBUG] Searching faculty...")
        
        faculty_members = session.query(models.Faculty).all()
        
        if not faculty_members:
            print(f"[DEBUG] No faculty in database")
        else:
            print(f"[DEBUG] Found {len(faculty_members)} faculty members")
            
            matched_faculty = None
            best_match_count = 0
            
            for faculty in faculty_members:
                faculty_name_lower = faculty.name.lower()
                
                # Count how many potential names match
                match_count = 0
                for potential_name in potential_names:
                    if potential_name in faculty_name_lower:
                        match_count += 1
                        print(f"[DEBUG] '{potential_name}' found in '{faculty.name}'")
                
                # Keep track of best match
                if match_count > best_match_count:
                    best_match_count = match_count
                    matched_faculty = faculty
            
            if matched_faculty:
                print(f"[DEBUG] Best match: {matched_faculty.name} ({best_match_count} matches)")
                
                return (
                    f"👨‍🏫 **{matched_faculty.name}** (Faculty)\n"
                    f"📞 Phone: {matched_faculty.phone}\n"
                    f"📧 Email: {matched_faculty.email}\n"
                    f"🏢 Department: {matched_faculty.department or 'N/A'}"
                )
            else:
                print(f"[DEBUG] No faculty found matching: {potential_names}")
    
    # ═══════════════════════════════════════════════════════════════
    # SEARCH WARDEN
    # ═══════════════════════════════════════════════════════════════
    
    if any(keyword in text_lower for keyword in ['warden', 'hostel']):
        print(f"[DEBUG] Searching wardens...")
        
        wardens = session.query(models.Warden).all()
        
        if not wardens:
            print(f"[DEBUG] No wardens in database")
        else:
            print(f"[DEBUG] Found {len(wardens)} wardens")
            
            matched_warden = None
            best_match_count = 0
            
            for warden in wardens:
                warden_name_lower = warden.name.lower()
                
                # Count how many potential names match
                match_count = 0
                for potential_name in potential_names:
                    if potential_name in warden_name_lower:
                        match_count += 1
                        print(f"[DEBUG] '{potential_name}' found in '{warden.name}'")
                
                # Keep track of best match
                if match_count > best_match_count:
                    best_match_count = match_count
                    matched_warden = warden
            
            if matched_warden and best_match_count > 0:
                print(f"[DEBUG] Best match: {matched_warden.name} ({best_match_count} matches)")
                
                return (
                    f"🏠 **{matched_warden.name}** (Warden)\n"
                    f"📞 Phone: {matched_warden.phone}\n"
                    f"📧 Email: {matched_warden.email}\n"
                    f"🏢 Hostel: {matched_warden.hostel_name or 'N/A'}"
                )
            
            elif len(potential_names) == 0 and wardens:
                # Generic query
                matched_warden = wardens[0]
                print(f"[DEBUG] Generic warden query, returning: {matched_warden.name}")
                
                return (
                    f"🏠 **{matched_warden.name}** (Warden)\n"
                    f"📞 Phone: {matched_warden.phone}\n"
                    f"📧 Email: {matched_warden.email}\n"
                    f"🏢 Hostel: {matched_warden.hostel_name or 'N/A'}"
                )
            
            else:
                print(f"[DEBUG] No warden found matching: {potential_names}")
    
    # ═══════════════════════════════════════════════════════════════
    # SEARCH WARDEN (existing code - keep as is)
    # ═══════════════════════════════════════════════════════════════
    
    if any(keyword in text_lower for keyword in ['warden', 'hostel']):
        print(f"[DEBUG] Searching wardens...")
        # ... existing warden search code ...
    
    print(f"[DEBUG] No contact information found")
    return None


def try_get_location(text: str, session) -> Optional[str]:
    words = text.lower().split()
    results = []
    
    for word in words:
        if len(word) > 2:
            # ===============================================================
            # Search in ROOM table
            # ===============================================================
            stmt = select(models.Room).where(
                models.Room.room_no.ilike(f"%{word}%") |
                models.Room.building.ilike(f"%{word}%")
            )
            room_rows = session.execute(stmt).all()
            
            for row in room_rows:
                r = row[0]
                data = f"🚪 **Room {r.room_no}**\n"
                if r.building:
                    data += f"🏢 Building: {r.building}\n"
                if r.floor:
                    data += f"🏗️ Floor: {r.floor}\n"
                if r.map_link:
                    data += f"🗺️ Map: {r.map_link}\n"
                results.append(data)
            
            # ===============================================================
            # Search in BUILDING table
            # ===============================================================
            stmt = select(models.Building).where(
                models.Building.name.ilike(f"%{word}%") |
                models.Building.code.ilike(f"%{word}%")
            )
            building_rows = session.execute(stmt).all()
            
            for row in building_rows:
                b = row[0]
                data = f"🏢 **{b.name}** ({b.code if b.code else 'Building'})\n"
                if b.address:
                    data += f"📍 Address: {b.address}\n"
                if b.lat and b.lng:
                    data += f"🌐 Coordinates: {b.lat}, {b.lng}\n"
                    data += f"🗺️ Google Maps: https://maps.google.com/?q={b.lat},{b.lng}\n"
                results.append(data)
    
    # Return combined results or None
    if results:
        return "\n---\n".join(results)
    return None


def try_get_faculty(text: str, session) -> Optional[str]:
    words = text.lower().split()
    results = []
    
    for word in words:
        if len(word) > 3:
            # Search by name OR department
            stmt = select(models.Faculty).where(
                models.Faculty.name.ilike(f"%{word}%") |
                models.Faculty.department.ilike(f"%{word}%")
            )
            rows = session.execute(stmt).all()
            
            for row in rows:
                f = row[0]
                data = f"👨‍🏫 **{f.name}**\n"
                data += f"🏢 Department: {f.department}\n"
                data += f"🚪 Office: {f.office_location}\n"
                if f.phone:
                    data += f"📞 Phone: {f.phone}\n"
                if f.email:
                    data += f"📧 Email: {f.email}\n"
                results.append(data)
    
    # Return combined results or None
    if results:
        # Remove duplicates (same faculty found by multiple words)
        unique_results = list(dict.fromkeys(results))
        return "\n---\n".join(unique_results)
    return None


def try_get_rag(text: str) -> Optional[str]:
    try:
        print(f"[DEBUG] Querying RAG with: {text}")
        
        from core.rag import get_rag_system
        rag_system = get_rag_system()
        
        # Check if RAG system is initialized
        stats = rag_system.get_collection_stats()
        if stats['count'] == 0:
            print("[DEBUG] ChromaDB is empty. Run: python scripts/ingest_pdfs.py")
            return None
        
        print(f"[DEBUG] ChromaDB has {stats['count']} documents")
        
        # Search using ChromaDB
        documents = rag_system.search_documents(text, top_k=3, min_score=0.1)
        
        print(f"[DEBUG] RAG returned {len(documents)} results")
        
        if not documents:
            print("[DEBUG] No relevant documents found")
            return None
        
        snippets = []
        MAX_CHARS_PER_CHUNK = 500  # ← ADD THIS (adjust as needed)
        
        for i, doc in enumerate(documents, 1):
            score = doc.get('relevance_score', 0)
            print(f"[DEBUG] Result {i}: score={score:.3f}")
            
            content = doc.get('content', 'No content')
            
            # TRUNCATE EACH CHUNK
            truncated_content = content[:MAX_CHARS_PER_CHUNK]
            if len(content) > MAX_CHARS_PER_CHUNK:
                truncated_content += "..."
            
            snippets.append(f"[Relevance: {score:.2f}]\n{truncated_content}")
        
        print(f"[DEBUG] Returning {len(snippets)} snippets")
        
        # Return truncated data with separator
        raw_data = "\n\n---\n\n".join(snippets)
        return raw_data
        
    except Exception as e:
        print(f"[ERROR] RAG error: {e}")
        import traceback
        traceback.print_exc()
        return None


def handle_small_talk(text: str) -> ChatResponse:
    """Handle greetings"""
    greetings = {
        "hi": "Hi there!",
        "hello": "Hello!",
        "hey": "Hey!",
        "good morning": "Good morning!",
        "good afternoon": "Good afternoon!",
        "good evening": "Good evening!",
    }
    
    text_lower = text.lower()
    for greeting_word, response in greetings.items():
        if greeting_word in text_lower:
            return ChatResponse(
                answer=(
                    f"{response}\n\n"
                    "I'm **Campus Companion** — your AI assistant for:\n"
                    "📞 Contact information (Canteens, Faculty, Wardens)\n"
                    "📍 Location (Rooms, Buildings)\n"
                    "👨‍🏫 Faculty information\n"
                    "📚 Academic rules & policies\n\n"
                    "What can I help you with?"
                ),
                intent="small_talk",
                confidence=0.95,
                used_fallback=False
            )
    
    return ChatResponse(
        answer=(
            "Hi! 👋 I'm **Campus Companion**.\n\n"
            "Ask me about:\n"
            "• Canteen contacts 🍽️\n"
            "• Faculty info 👨‍🏫\n"
            "• Room locations 🚪\n"
            "• Warden contacts 🏠\n"
            "• CGPA rules 📚\n"
            "• And more!"
        ),
        intent="small_talk",
        confidence=0.9,
        used_fallback=False
    )





