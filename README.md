# 🎓 Campus Companion

**AI-Powered Campus Information Assistant for NIT Durgapur**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend API** | FastAPI + Uvicorn |
| **Database** | SQLite3 |
| **Vector Storage** | ChromaDB (internally using SQLite3) |
| **Embeddings** | Sentence Transformers → 384-dim vectors |
| **Frontend** | Streamlit |
| **PDF Loading** | PyPDF Loader |
| **Classification** | Keyword → Logistic Regression (ML) → LLM |
| **LLM** | Open Source Model from HuggingFace: Mistral-7B-Instruct |

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CAMPUS COMPANION SYSTEM                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USER QUERY → FastAPI Backend → 3-Level Classification           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  INTENT CLASSIFIER (core/classifier.py)                    │  │
│  │                                                            │  │
│  │  Level 1: Keyword Matching (⚡ 0.001s) - 70% queries        │  │
│  │  Level 2: ML Classifier (⚡⚡ 0.01s) - 25% queries           │  │
│  │  Level 3: LLM (Mistral-7B) (⚡⚡⚡ 1-2s) - 5% queries         │  │
│  └──────────────────────┬─────────────────────────────────────┘  │
│                         │                                        │
│         ┌───────────────┼───────────────┐                        │
│         ▼               ▼               ▼                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
│  │ DATABASE   │  │ RAG SYSTEM │  │ AI FALLBACK│                  │
│  │            │  │            │  │            │                  │ 
│  │ • Canteen  │  │ • ChromaDB │  │ Mistral-7B │                  │
│  │ • Faculty  │  │ • 384-dim  │  │ Generates  │                  │
│  │ • Rooms    │  │   Vectors  │  │ Responses  │                  │
│  │ • Wardens  │  │ • Cosine   │  │            │                  │
│  │ (SQLite)   │  │   Search   │  │            │                  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                  │
│        │               │               │                         │
│        └───────────────┼───────────────┘                         │
│                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  AI RESPONSE FORMATTER (core/response.py)                  │  │
│  │  Raw Data → Natural Language (Mistral-7B, Temp: 0.5)       │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  JSON RESPONSE                                             │  │
│  │  {"answer": "...", "intent": "...", "confidence": 0.85}    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CAMPUS_COMPANION/
├── api/
│   ├── main.py                    # FastAPI app initialization
│   └── routers/
│       └── chat.py                # ⭐ Main chat endpoint (600+ lines)
├── core/
│   ├── classifier.py              # ⭐ 3-level intent classification (800+ lines)
│   ├── rag.py                     # ⭐ RAG system with ChromaDB (190+ lines)
│   ├── response.py                # 🤖 AI response formatter
│   ├── fallback_message.py        # 🛡️ AI fallback handler
│   └── embeddings.py              # Document chunking & embeddings
├── db/
│   ├── models.py                  # ⭐ Database schema (10 tables)
│   └── session.py                 # DB connection
├── scripts/
│   ├── ingest_pdfs.py             # PDF → ChromaDB pipeline
│   ├── pdf_processor.py           # Text extraction (PyPDF2 + Tesseract)
│   └── chunking.py                # Text chunking logic
├── data/
│   ├── pdfs/                      # Source PDF documents
│   └── rag_docs/                  # ChromaDB storage
├── frontend.py                    # Streamlit chat UI
├── app.py                         # Database initializer
├── testdb.py                      # Sample data loader
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
└── campus_companion.db            # SQLite database
```

---

## 🚀 Quick Start Guide

### �� Prerequisites

Verify the following are installed:

```bash
python3 --version    
pip --version
git --version
```

### 🔧 Installation

**1. Clone and Navigate**
```bash
git clone <your-repo-url>
cd CAMPUS_COMPANION
```

**2. Create Virtual Environment**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up HuggingFace Token**
1. Visit: https://huggingface.co/settings/tokens
2. Create token (Read access)
3. Copy token
4. Create `.env` file:
```bash
echo "HUGGINGFACEHUB_API_TOKEN=hf_paste_your_token_here" > .env
```

**5. Initialize Database**
```bash
python3 app.py
python3 testdb.py
```

**6. Set Up PDF Documents**
```bash
mkdir -p data/pdfs
# Add your PDF documents to data/pdfs/
# Then run:
python3 scripts/ingest_pdfs.py
```

**7. Start Backend**
```bash
uvicorn api.main:app --reload
```

**8. Test API** (in new terminal)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"hello"}'
```

**9. Start Frontend** (in another terminal with `.venv` activated)
```bash
streamlit run frontend.py
```

---

## 📚 Implementation Guide

### 📅 DAY 1: Database Setup & AI Fallback

**🎯 Learning Objectives:**
- Understand the 3-level intent classification system
- Learn database structure and queries
- Implement AI fallback for graceful error handling

#### 🔍 The Problem
How does the system know what type of question was asked?

**Solution:** Progressive complexity with fallback

#### 🏆 Three-Level Classification System

##### **Level 1: Keyword Matching** ⚡ (Fast - 0.001s)
- **Handles:** 70% of queries
- **Method:** Simple word detection
- **Function:** `classify_keyword()` in `classifier.py`

**Examples:**
- ✅ "Roy canteen phone" → Keywords found → `db_contact`
- ✅ "Where is AB-301?" → Keywords found → `db_location`
- ❌ "I need to contact the mess" → No exact keywords

##### **Level 2: Machine Learning** ⚡⚡ (Medium - 0.01s)
- **Handles:** 25% of queries
- **Method:** TF-IDF + Logistic Regression
- **Training:** Pre-trained on 200+ example queries
- **Function:** `ml_classify()` in `classifier.py`

**How it works:**
- Converts text to numerical features (word importance)
- Trained model predicts intent
- Example: "mess contact" → ML recognizes as contact query

**When it works:**
- ✅ Variations of known patterns
- ✅ Synonyms and paraphrases

**When it fails:**
- ❌ Completely novel phrasing
- ❌ Ambiguous questions

##### **Level 3: LLM Classification** ⚡⚡⚡ (Slow ~ 1-2s) [HW]

**Hints:**
- Sends query to Mistral-7B with instructions
- "Classify this as: contact/location/rag/small_talk/fallback"
- Returns intent with reasoning

**Example:**
- "Can you help me reach the person in charge of food services?" → LLM understands context → `db_contact`

#### 🗄️ Database Structure

**How to create the DB:**
1. Create table models in `models.py`
2. Use `session.py` to connect
3. Populate db by forming `testdb.py`

**What's in the Database?**
- 8-10 tables in SQLite: Faculty, Canteen, Warden, Room, Building, etc.
- Fixed schema (columns known in advance)
- Fast exact matches

**Key Functions:**
- `try_get_contact(text, session)` - Search people/places
- `try_get_location(text, session)` - Search rooms/buildings
- `extract_entity_names()` - Parse query for names

#### 🛡️ AI Fallback System

**Concept:** Graceful handling of out-of-scope queries

**When Fallback Triggers:**
- Intent classified as "ai_fallback"
- Database search returns nothing
- RAG search finds no relevant documents
- Confidence too low (< 0.3)

**Response Generation:**
- Send query + system prompt to Mistral-7B
- Temperature: 0.7 (more creative for deflection)
- AI generates polite refusal + redirection + organized response

**Key Function:** `fallback_ai_response(query)` in `fallback_message.py`

#### 🧪 Hands-on Exercise

Edit `testdb.py` and add/populate data of your choice

**Functions Involved:**
- `session.add()` - Add to database
- `session.commit()` - Save changes
- `try_get_contact()` - Search function

---

### 📅 DAY 2: RAG (Retrieval Augmented Generation)

**🎯 Learning Objectives:**
- Understand RAG (Retrieval Augmented Generation) concept
- Learn PDF processing pipeline (extraction → chunking → embedding)
- Understand semantic search and cosine similarity

#### 🤔 The Problem

**Student asks:** "How to calculate CGPA?"

**Why Database Won't Work:**
- ❌ CGPA calculation is a multi-step explanation (not a single data point)
- ❌ Rules are in PDF documents (Academic Handbook, 50+ pages)
- ❌ Manual data entry = tedious + error-prone
- ❌ Rules change → Need to update database every time

**Why not raw AI?**
- ❌ LLMs hallucinate
- ❌ No existing knowledge of the campus rules
- ❌ CGPA varies from campus to campus

#### 🔄 RAG Pipeline

**Document Processing:**
```
PDFs → Extract Text → Split into Chunks → Convert to Vectors → Store in Database
```

**Query Processing:**
```
User Question → Convert to Vector → Find Similar Vectors → Get Text Chunks
Question + Context Chunks → LLM → Natural Answer
```

#### 📄 PDF Processing Pipeline

##### **1. PDF Text Extraction**

**Process:**
1. Open PDF file
2. Iterate through each page
3. Extract text layer (embedded text data)
4. Concatenate all pages

**Key Function:** `extract_text_pypdf2()` in `pdf_processor.py`

##### **2. Text Cleaning**

**Removes Noise:**
- Page numbers
- URLs
- Headers/footers
- Extra whitespaces

**Key Function:** `clean_text()` in `pdf_processor.py`

##### **3. Quality Check**

Check the length of answer and whether it is in readable format

**Key Function:** `validate_extracted_text()` in `pdf_processor.py`

#### ✂️ Text Chunking

**Why Chunking?**
- Embedding models have token limits for each query
- Semantic search less accurate with more tokens
- Higher API cost with larger inputs

**Solution:** Split into smaller, semantically meaningful pieces

**Chunking Parameters:**
- `chunk_size`
- `chunk_overlap`
- `min_chunk_size`

**Key Function:** `chunk_text()` in `chunking.py`

#### 🔢 Embeddings

**Model Used:** `all-MiniLM-L6-v2` (Sentence Transformers)

**Concept:** Convert text into numbers that capture meaning

**Example:**
```
Text: "How to calculate CGPA?"
Embedding: [0.234, -0.112, 0.567, ..., 0.891]  (384 numbers)

"CGPA calculation" → [0.12, 0.45, -0.23, ...]
"Grade point average" → [0.15, 0.43, -0.20, ...]  (CLOSE! ✅)
"Pizza recipe" → [0.87, -0.32, 0.61, ...]  (FAR! ❌)
```

**Key Functions:**
- `get_embeddings()` in `embeddings.py`
- `generate_embeddings()` in `embeddings.py`

#### 🗃️ Vector Database - ChromaDB

Stores all the embeddings for semantic search

#### 🧪 Hands-on Exercise

```bash
# 1. Add PDF to data folder
cp ~/hostel_rules.pdf data/pdfs/

# 2. Run ingestion
python3 scripts/ingest_pdfs.py

# 3. Check ChromaDB
python3 -c "from core.rag import collection; print(f'{collection.count()} documents')"

# 4. Test query
curl -X POST http://localhost:8000/api/chat \
  -d '{"text":"What are hostel visiting hours?"}'
```

---

### 📅 DAY 3: Intent Classifier

**🎯 Learning Objectives:**
- Master the unified classification system
- Understand the priority-based keyword matching
- Learn ML-based intent prediction
- Explore result aggregation strategies

#### 🤔 The Problem

When a user asks "Roy canteen phone", how does the system know they want contact information and not location or rules?

**Solution:** Intent Classification - categorizing user queries into predefined intents

#### 🎯 Intent Types in Campus Companion

| Intent | Description |
|--------|-------------|
| `db_contact` | Contact information (phone, email) |
| `db_location` | Location queries (rooms, buildings) |
| `rag` | Document-based questions (CGPA rules, policies) |
| `ai_fallback` | General questions / greetings |
| `small_talk` | [HW] Conversational queries |

#### 🔄 Three-Level Classification Pipeline

```
Keyword Matching (Fast) → Machine Learning (Accurate) → LLM (Slow but most Accurate) [HW]
```

#### 🔑 Keyword Classification

**Function:** `classify_keywords(text: str) -> IntentResult`

**Purpose:** Fast rule-based classification using keyword matching

**Priority Order (Matters!):**
1. ✅ Check for RAG keywords → "CGPA", "rules", "policy"
2. ✅ Check for contact keywords → "phone", "email", "canteen"
3. ✅ Check for location keywords → "where", "room", "building"
4. ✅ Default → `ai_fallback`

**Why this order?**
- RAG first because academic queries are most specific
- Contact/Location second because they have clear entities
- Fallback last as catch-all

#### 🤖 Machine Learning Classifier

**Key Class:** `MLClassifier`

**Purpose:** Learn patterns from training examples using Machine Learning

**Components:**
1. **TF-IDF Vectorizer** - Converts text to numerical features
2. **Logistic Regression** - Predicts intent based on learned patterns

#### 🎼 The Orchestrator

**Function:** `UnifiedClassifier.classify()`

**Purpose:** Combine all three classifiers and make final decision

**Classification Pipeline:**
```
Step 1: Run keyword classifier (always)
  ↓
Step 2: Run ML classifier (if trained)
  ↓
Step 3: Run LLM classifier (if requested AND confidence < 0.7)
  ↓
Step 4: Aggregate results by taking MAX confidence per intent
  ↓
Step 5: Detect multi-intent queries
  ↓
Step 6: Determine if AI fallback needed
  ↓
Return ClassificationResult
```

#### 📊 Result Aggregation Strategy

**Why MAX (not AVG)?**

- If one classifier is very confident, it likely found a strong signal
- Average would dilute strong predictions
- **Example:** Keyword (0.90) + ML (0.60) → MAX = 0.90 (better than AVG = 0.75)

**[HW]** Multi-intent Discussion

---

### 📅 DAY 4: Response Generation + Frontend

**🎯 Learning Objectives:**
- Understand how raw data is converted to natural language responses
- Learn the role of AI in response formatting
- Understand the frontend-backend connection

#### 🤔 The Problem

**Database returns raw data for "Roy canteen phone":**

```
Raw Output:
name: Roy Canteen
phone: +91-8012345678
email: roy@campus.edu
location: Ground Floor
```

- **User Experience:** ❌ Boring, mechanical, not conversational
- **What Users Expect:** ✅ Natural, helpful, human-like response

#### 💡 The Solution: AI Response Formatter

**Output:**
```
🍽️ Roy Canteen

You can reach Roy Canteen at +91-8012345678 or email them at 
roy@campus.edu. They're located on the Ground Floor!
```

#### 🔄 Response Flow

```
RAW DATA (from DB/RAG)
    ↓
AI FORMATTER (response.py)
    ↓
NATURAL LANGUAGE RESPONSE
    ↓
FRONTEND (frontend.py)
    ↓
USER SEES POLISHED ANSWER
```

#### 🤖 AI Response Formatter Architecture

**Key Class:** `ResponseGenerator` in `response.py`

##### **Main Methods:**

**1. `__init__()` - Initialization**
- Purpose: Set up LLM (Mistral-7B) and RAG system

**2. `refine_query(query: str) -> str`**
- Purpose: Improve search queries before RAG lookup

**3. `format_response(query: str, data: str) -> str`**
- Purpose: Convert raw data to natural language

**4. `generate_rag_response(query: str) -> Dict`**
- Purpose: Complete RAG pipeline - search docs + generate answer

##### **Helper Functions:**

**`_build_context(documents, max_length)`**
- Combines document chunks into one string
- Stops at 2000 chars (LLM context limit)
- Labels each source: [Source 1], [Source 2], etc.

**`_generate_llm_answer(query, context)`**
- Sends context + query to Mistral-7B
- Prompt engineering: "Answer using ONLY context"
- Prevents hallucination (AI making up facts)

**`_calculate_confidence(documents)`**
- Average relevance score of top 3 chunks
- Example: (0.92 + 0.87 + 0.81) / 3 = 0.87

**`_format_sources(documents)`**
- Extract metadata: filename, relevance score
- Show users where answer came from (transparency)

**5. `_generate_contact_response(query: str) -> Dict`**
- Purpose: Format database contact results

**6. `_generate_location_response(query: str) -> Dict`**
- Purpose: Format database location results

**7. `_generate_ai_fallback_response(query: str) -> Dict`**
- Purpose: Handle out-of-scope queries gracefully

**8. `generate_response(query: str, intent: str) -> Dict`**
- Purpose: Main entry point - routes to correct handler

#### 🖥️ Frontend - Streamlit

**What is Streamlit?**
- Streamlit = Python web framework for data apps

**Why Streamlit?**
- ✅ Write web UI in pure Python (no HTML/CSS/JavaScript)
- ✅ Auto-refreshes on code changes
- ✅ Built-in chat components (`st.chat_message`, `st.chat_input`)
- ✅ Fast prototyping (build UI in 50 lines!)

##### **Key Components:**

1. **Page Configuration:** `st.set_page_config`
2. **Sidebar:** `st.sidebar`
3. **Session State:** Conversation memory and Chat History [HW]
4. **Chat Input & API Call:** `st.chat_input`

#### 🚀 Running the Application

```bash
# Start backend
uvicorn api.main:app --reload

# Start frontend (on another terminal with .venv activated)
streamlit run frontend.py
```

#### ❓ Common Questions

**Q: Why separate frontend and backend?**
- A: Scalability. Backend can serve multiple frontends (web, mobile, API users).

**Q: Can we use React instead of Streamlit?**
- A: Yes! Backend API is framework-agnostic. Just POST to `/api/chat`.

**Q: Why not format in chat.py directly?**
- A: Separation of concerns. `response.py` is reusable across different endpoints.

**Q: How to deploy to production?**
- A: Backend → Railway/Render. Frontend → Streamlit Cloud (free tier).

---

### 📅 DAY 5+6: Chat System + FastAPI

**🎯 Learning Objectives:**
- Understand FastAPI application structure
- Learn request/response flow
- Master the chat endpoint orchestration

#### 📄 main.py - The Entry Point

**Purpose:** Entry point of the backend

**Key Idea:**
- Nothing intelligent happens here
- It does not answer questions
- It sets up everything needed so other files can work

##### **Key Components:**

**1. `app = FastAPI(...)`**
- App is the control center of the backend
- Every endpoint, rule, and config is attached to it

**2. CORS Middleware Block**
- Frontend and backend usually run on different ports
- Browsers block such requests by default

**CORS Configuration:**
- `allow_origins` → Who can access the backend
- `allow_methods` → What HTTP actions are allowed
- `allow_headers` → What headers are accepted
- `allow_credentials` → Whether cookies/auth can pass

**3. `init_db()`**
- Database tables exist before any request
- Backend never crashes due to missing tables
- Reads database models
- Creates tables if missing
- Skips if already present

**4. `app.include_router(...)`**
- A router is a group of related endpoints
- Example: chat routes live in `chat.py`
- Connects `/api/chat` → logic in `chat.py`
- Adds structure and modularity

**5. Root Endpoint `/`**
- Helpful for debugging, deployment checks, dev sanity checks

**6. Health Check `/health`**
- Every production backend has a health endpoint
- It answers only one thing: "Am I alive?"

#### 📄 chat.py - The Orchestrator

**Purpose:** Where user input becomes an intelligent response

**Responsibilities:**
- Receiving user queries
- Validating input
- Classifying intent
- Fetching data (DB / RAG)
- Using AI when needed
- Returning a structured response

##### **Chat Endpoint: `/api/chat`**

**Role:**
- Single entry point for all user queries

**Handles:**
- Simple greetings
- Database lookups
- Document-based questions
- AI fallback responses

**Why one endpoint?**
- Simplifies frontend
- Centralizes logic
- Easier to debug and extend

##### **Key Function:** `chat(request: ChatRequest)`

**Explanation:**
- This function is the orchestrator — it doesn't do everything itself, but controls everything

##### **Request & Response Models**

**`ChatRequest`**

**Purpose:**
- Guarantees valid input
- Prevents malformed data
- Makes API predictable

**`ChatResponse`**

**Purpose:**
- Standardizes backend output
- Makes frontend rendering easy

**Fields:**
- `answer` → Final message
- `intent` → What the system understood
- `confidence` → How sure the system is
- `used_fallback` → Whether AI was used
- `is_multi_intent` → Multiple meanings detected
- `all_intents` → Ranked intent candidates

##### **Intent Classification Pipeline**

**Key Function:** `classify_detailed`

**Purpose:**
- The system decides what the user wants, not how to answer yet

**Types of intents:**
- `db_contact`
- `db_location`
- `faculty_info`
- `rag`
- `small_talk` [HW][greetings]
- `ai_fallback`

**Why classification first?**
- Avoids unnecessary DB calls
- Prevents wrong answers

**Important classification outputs:**
- `primary_intent`
- `confidence`
- `needs_fallback`
- `is_multi_intent`
- `all_intents`

##### **Handlers & Data Retrieval**

**Main routing decision:** Based on `primary_intent`

**Important Handler Functions:**
- `try_get_contact()` - Search for contact information
- `try_get_location()` - Search for locations
- `try_get_faculty()` - Search for faculty information
- `try_get_rag()` - Retrieve from RAG system
- `fallback_ai_response()` - Handle unknown queries

##### **Response Formatting**

Final step before your query is sent, processed through a number of steps and ready to be printed in JSON format → formatting is required to return in user-friendly form

---

## 🎓 Course Summary

### Dear Students,

Over the past 6 days, we built **Campus Companion**, an AI-powered chatbot that helps students find contact information, locations, and academic policies through a beautiful Streamlit interface. 

#### 🏗️ System Architecture

The system uses a **3-layer architecture**:
1. **Frontend** (Streamlit for UI)
2. **Backend** (FastAPI for API server)
3. **Core Intelligence** (classification, database handlers, RAG, and AI formatting)

#### 🔄 Request Flow

When a user asks "Roy canteen phone", the request flows through:
1. **Pydantic validation**
2. **3-level intent classification** (keywords/ML/LLM)
3. **Routing to appropriate handler** (`try_get_contact` searches the database with fuzzy matching)
4. **AI formatting** (Mistral-7B converts raw data to natural language)
5. **Structured JSON response** displayed in the frontend

#### 🛠️ Technologies Used

**Modern Stack:**
- **FastAPI** (REST API)
- **SQLAlchemy** (database ORM)
- **Scikit-learn** (ML classification)
- **ChromaDB** (vector database for RAG)
- **HuggingFace** (embeddings and LLM)

**Production-Grade Principles:**
- ✅ Separation of concerns
- ✅ Graceful degradation (fallback mechanisms)
- ✅ Comprehensive error handling
- ✅ Type safety with Pydantic
- ✅ Extensive logging

#### 💡 Key Innovation

Our **hybrid approach** combines:
- **Structured database queries** for contacts/locations
- **RAG (Retrieval-Augmented Generation)** for document-based questions like "How to calculate CGPA?"
  - Uses semantic search to find relevant PDF chunks
  - Generates contextual answers

#### 🎯 What You've Learned

1. **Full-stack development** (frontend + backend + database)
2. **AI/ML integration** (classification, embeddings, LLMs)
3. **Software engineering** (clean architecture, error handling, API design)
4. **Real-world application** that solves actual campus problems

#### 🚀 Real-World Applications

This same architecture can be adapted for:
- 🏥 Hospital assistants
- 🏢 Corporate helpdesks
- 🛒 E-commerce support
- 📚 Any domain requiring intelligent information retrieval

#### 🔧 Next Steps

You're now ready to:
- **Extend** this system (add new intents, multilingual support)
- **Improve** accuracy (fine-tune classifiers, better RAG strategies)
- **Deploy** to production (Railway/Render/AWS)
- **Add** advanced features (voice input, analytics dashboards)

#### 🎉 Congratulations!

You didn't just learn to code, you learned to **think like a software engineer**, understanding:
- Why each component exists
- How they communicate
- When to use different approaches

These are skills that companies actively seek in full-stack AI developers.

### **Now go build something amazing! 🚀**

---

## 🏆 Skills Mastered

**FastAPI** + **Streamlit** + **SQLAlchemy** + **ChromaDB** + **HuggingFace** + **RAG** + **Clean Architecture**

**Keep coding, keep learning, keep building! 💙**

---

## 📞 Support

For questions or issues, please refer to the implementation guide above or contact the development team.

---

**Made with ❤️ for NIT Durgapur**
