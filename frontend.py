# =============================== DAY - 5 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     FRONTEND - STREAMLIT UI                              ║
║                    Campus Companion Chat Interface                       ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This is the USER INTERFACE for the Campus Companion chatbot.
It provides a web-based chat interface where students can ask questions.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                     CAMPUS COMPANION SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  👤 USER (Web Browser)                                              |
│      ↓                                                              │
│  [1] FRONTEND (THIS FILE) ← You are here!                           │
│      • Displays chat interface                                      │
│      • Sends user queries to backend                                │
│      ↓                                                              │
│  [2] API BACKEND (api/main.py + api/routers/chat.py)                │
│      • Receives query                                               │
│      • Routes to appropriate handler                                │
│      ↓                                                              │
│  [3] PROCESSING LAYER (core/)                                       │
│      • Classifies intent                                            │
│      • Searches database/RAG                                        │
│      • Generates response                                           │
│      ↓                                                              │
│  [4] DATA LAYER                                                     │
│      • SQLite Database (db/)                                        │
│      • ChromaDB Vector Store (data/rag_docs/)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────
1. Creates a chat interface using Streamlit framework
2. Displays conversation history
3. Sends user messages to backend API (http://localhost:8000/api/chat)
4. Shows bot responses with metadata (intent, confidence)
5. Handles errors gracefully (connection errors, timeouts)

🚀 HOW TO RUN:
─────────────────────────────────────────────────────────────────────────
Terminal 1 - Start Backend:
    uvicorn api.main:app --reload

Terminal 2 - Start Frontend (This File):
    streamlit run frontend.py

Then open: http://localhost:8501

📊 DATA FLOW EXAMPLE:
─────────────────────────────────────────────────────────────────────────
User types: "Roy canteen phone"
    │
    ↓ [This file sends POST request]
    │
Backend (api/routers/chat.py):
    ├─→ Classifies: "db_contact" (via core/classifier.py)
    ├─→ Searches Database: SELECT * FROM canteen WHERE name LIKE '%roy%'
    ├─→ Formats response: "Roy Canteen: 9876543210"
    │
    ↓ [Returns JSON response]
    │
This file displays: "Roy Canteen: 9876543210"

🔑 KEY FEATURES:
─────────────────────────────────────────────────────────────────────────
✓ Real-time chat interface
✓ Message history (stored in session)
✓ Clear chat button
✓ Loading spinner during API calls
✓ Error handling for network issues
✓ Metadata display (intent, confidence)
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════
import streamlit as st
import requests


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

set.set_page_config(
    page_title = "Campus Companion",
    layout = "centered"
)



# ═══════════════════════════════════════════════════════════════════════
# API ENDPOINT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════


API_URL = "http://127.0.0.1:8000/api/chat"


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR: Navigation and Information Panel
# ═══════════════════════════════════════════════════════════════════════


with st.sidebar:
    st.header("Campus Companion")
    st.markdown("Your AI assistant for campus life.")

    st.divider()

    if st.button("Clear Chat history",type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### What can I help with?")
    st.markdown(""" 
    - **Contacts:** Canteens,Faculty,Wardens
    - **Locations:** Rooms & Buildings
    - **Documents:** CGPA Rules, Policies
    - **General:** Questions & Greetings
    """)
    st.divider()

    st.caption("Powered by RAG + Mistral-7B AI")
# ═══════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════

st.title("Campus Companion Chat")
st.markdown("Ask Me about campus!")




# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []





# ═══════════════════════════════════════════════════════════════════════
# DISPLAY CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════════

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ex: 'Roy canteen phone' pr 'How to calculate CGPA?'"):
    with st.chat_message(message["user"]):
        st.markdown(message["prompt"])
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking...."):
            try:
                response = requests.post(API_URL,json = {"text":prompt},timeout = 30)
            if reponse.status_code = 200:
                data = response.json()

                bot_reply = data.get("answer","Sorry, I couldn't generate a response.")

                intent = data.get("intent","Unknown")
                confidence = data.get("confidence",0)
                used_fallback = data.get("used_fallback",False)

                metadata = f"\n\n---\n"
                metadata += f"*Intent: `{intent}` |"
                metadata +- f"Confidence: `{confidence:.2f}` |"
                metadata += f"AI Fallback: `{'Yes' if used_fallback else 'No'}`*"

                full_response = bot_reply + metadata
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({
                    "role":"assisant",
                    "content":"full_response"
                })
            else :
                error_msg = f"Server Error {response.status_code}"
                message_placeholder.error(error_msg)

        except requests.exceptions.ConnectionError:
            error_msg = (
                "Refused to connect"
            )
            message_placeholder.error(error_msg)

        except requests.exceptions.Timeout:
            error_msg = "TimeOut!"
            message_placeholder.error(error_msg)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.error(error_msg)


# ═══════════════════════════════════════════════════════════════════════
# CHAT INPUT & RESPONSE HANDLING
# ═══════════════════════════════════════════════════════════════════════






