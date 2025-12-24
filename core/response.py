# ================================ DAY - 5 ================================ #
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
import logging 
import os
from typing import Dict,List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage

from core.rag import get_rag_system
from core.classifier import classify_detailed

try:
  from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
  HAVE_HF_LLM = True 
except ImportError:
  HAVE_HF_LLM = False 
  printf("Warining: langchain-huggingface not installed")

  
from db.session import SessionLocal
from db import models

load_dotenv()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)






# =======================================================================
# RESPONSE GENERATOR CLASS
# =======================================================================
class ResponseGenerator:

  def __init__(self):
    self.rag = get_rag_system()
    self.llm = None

    if HAVE_HF_LLM:
      try:
        hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

        hf_llm = HuggingFaceEndpoint(
          repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
          max_new_tokens = 512,
          temperature = 0.3,
          huggingfacehub_api_token = hf_token,
          timeout = 120
        )

        self.llm = ChatHuggingFace(llm = hf_llm)
        logger.info("HuggingFace Mistral-7B initialized")
      except Exception as e:
        logger.warning(f"HF init failed: {e}")

    if not self.llm:
      logger.warning("No LLM available.")

#Query Refinement

  def refine_query(self,query:str) -> str:
    query = query.strip()
    if len(query.spilt()) <=3 or not self.llm:
      return query

    try:
      messages = [
        SystemMessage(content="Refine queries for semantic search."),
        HumanMessage(content =f"Refine this query:\n{query}")
      ]
      response = self.llm.invoke(messages)
      return response.content.strip().split("\n")[0]
    except Exception:
      return query

# Response Formatting
  def format_response(self,query:str,data:str) ->str:
    if not self.llm:
      return f"Here what I found:\n\n{data}"

    try:
      messages = [
        SystemMessage(content = "Format clearly and concisely."),
        HumanMessage(content=f"Question: {query}\n\nData:\n{data}")
      ]
      return self.llm.invoke(messages).content.strip()

    except Exception:
      return f"Here's what I found:\n\n{data}"

  #rag
  def generate_rag_response(
    self,
    query: str,
    max_context_length: int = 2000,
    top_k: int = 5
  ) -> Dict[str, any]:
    refined_query = self.refine_query(query)
    documents = self.rag.search_documents(refined_query,top_k = top_k)
    if not documents:
      return {
        "answer":"No relevant information found.",
        "source": [],
        "confidence":0.0,
        "method":"rag_no_results"
      }
    
    context = self._build_context(documents,max_context_length)

    if self.llm:
      answer = self._generate_llm_answer(query,context)
      method = "rag_hf_llm"
    else:
      answer = self._format_context_answer(documents)
      method = "rag_basic"

    return {
      "answer" : answer,
      "sources" : self._format_sources(documents),
      "confidence" : self._calculate_confidence(documents),
      "method" : method
    }

  def _build_context(self,documents: List[Dict], max_length: int) ->str:
    context = ""
    for i, doc in enumerate(documents, 1):
      block = f"[Source {i}]\n{doc['content']}\n\n"
      if len(context) + len(block) > max_length:
        break
      context += block
    return context

  def _generate_llm_answer(self,query:str,context:str) ->str:
    try:
      prompt: f"""Answer the question using ONLY the context below.
      Context : 
      {context}
      Question: {query}
      Answer:
      """
      messages = [
        SystemMessage(context = "Answer only from the context"),
        HumanMessage(content = prompt)
      ]
      return self.llm.invoke(messages).context.strip()
    except Exception:
      return self._format_context_answer([{"content" : context}])

  def _format_context_answer(self,documents:List[Dict]) ->str:
    parts = ["Based on available information"]
    for doc in documents[:3]:
      parts.append(doc["content"][:300] + "....")
    if len(documents) > 3:
      parts.append(f"...and {len(documents)-3} more sources")
    return "\n\n".join(parts)

  def _calculate_confidence(self,documents:List[Dict]) -> float:
    scores = [doc.get("revelant_score",0.5) for doc in documents[:3]]
    return round(sum(scores)/len(scores),2)

  def _format_sources(self,documents:List[Dict]) -> List[Dict]:
    return[
      {
        "filename" : doc["metadata"].get("filename","Unknown"),
        "relevance" : round(doc.get("relevance_score",0.0),2)
      }
    ]

  def generate_response(self,query:str, intent:str = None) -> Dict[str,any]:
    if not intent:
      intent = classify_detailed(query).primary_intent

    if intent == "rag":
      return self.generate_rag_response(query)
    if intent == "db_contact":
      return self._generate_contact_response(query)
    if intent == "db_location":
      return self._generate_location_response(query)

    return self._generate_ai_fallback_response(query)

  def _generate_contact_response(self, query:str) -> Dict[str,any]:
    session = SessionLocal()
    try:
      from sqlmodel import select 
      canteens = session.excecute(select(models.Canteen)).all()

      if not canteens:
        return {"answer":"No contact information found.","sources":[],"confidence":0.0}
      
      c = canteens[0][0]
      answer = f"{c.name}\n"
      answer += f"Phone: {c.phone}\n"
      if c.email:
        answer += f"Email : {c.email}"

      return {
        "answer" : answer,
        "sources" : [{"table":"cateen"}],
        "confidence" : 0.95
      }
    except Exception as e:
      logger.error(f"Content query error: {e}")
      return {"answer" : str(e),"source":[],"confidence":0.0}
    finally:
      session.close()

  def _generate_location_response(self,query:str) ->Dict[str,any]:
    session = SessionLocal()
    try:
      from sqlmodel import select
      rooms = session.execute(select(models.Room)).all()
      if not rooms:
        return {"answer":"No location info found.","source":[],"confidence":0.0}

      room = rooms[0][0]
      answer = f"Room {room.room_no}\n"
      if room.building:
        answer += f"Building: {room.building}\n"
      if room.floor:
        answer += f"Floor : {room.floor}"

      return {
        "answer":answer,
        "sources":[{"table" : "room"}],
        "confindence" : 0.95
      }
    except Exception as e:
      logger.eroor(f"Location query error: {e}")
      return {"answer":str(e), "source" :[],"confidence":0.0}
    finally:
      session.close()

  def _generate_ai_fallback_response(self,query:str) ->Dict[str, any]:
    return{
      "answer":"I can help with academics, contacts and campus location",
      "sources":[],
      "confidence" : 0.6
    }

# --------------------------------------------------------------------
# GLOBAL HELPERS
# ----------------------------------------------------------------------

_response_generator = None

def get_repsonse_generator() -> ResponseGenerator:
  global _response_generator 
  if _response_generator is None:
    _response_generator = ResponseGenerator()
  return _response_generator

def generate_response (query:str, intent:str=None) -> Dict [str,any]:
  return get_repsonse_generator().generate_response(query,intent)

def generate_rag_response (query:str) -> Dict[str, any]:
  return get_repsonse_generator().generate_rag_response(query)

def format_response (query:str, data:str) ->str:
  return get_repsonse_generator().format_response(query,data)