# ================================ DAY - 5 ================================ #
import streamlit as st
import requests

# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Campus Companion",  # Browser tab title
    page_icon="🎓",                  # Browser tab icon
    layout="centered"                # Content centered (vs "wide")
)

# ═══════════════════════════════════════════════════════════════════════
# API ENDPOINT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
API_URL = "http://127.0.0.1:8000/api/chat"

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR: Navigation and Information Panel
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    # App header with icon
    st.header("🎓 Campus Companion")
    st.markdown("Your AI assistant for campus life.")
    
    # Visual separator
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════
    # CLEAR CHAT HISTORY BUTTON
    # ═══════════════════════════════════════════════════════════════════
    if st.button("🗑️ Clear Chat History", type="primary"):
        st.session_state.messages = []  # Reset message list
        st.rerun()                       # Refresh page to show empty chat

    # ═══════════════════════════════════════════════════════════════════
    # FEATURE SHOWCASE
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### What I Can Help With")
    st.markdown("""
    - 📞 **Contacts:** Canteens, Faculty, Wardens
    - 📍 **Locations:** Rooms & Buildings
    - 📚 **Documents:** CGPA Rules, Policies
    - 💬 **General:** Questions & Greetings
    """)
    
    st.divider()
    
    # Technical attribution (shows underlying technology)
    st.caption("Powered by RAG + Mistral-7B AI")

# ═══════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════
# This section creates the main content area where the chat happens

# Page title and description
st.title("🎓 Campus Companion Chat")
st.markdown("Ask me anything about campus!")

# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []  # Start with empty conversation

# ═══════════════════════════════════════════════════════════════════════
# DISPLAY CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════════
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Creates message bubble
        st.markdown(message["content"])      # Renders message text

# ═══════════════════════════════════════════════════════════════════════
# CHAT INPUT & RESPONSE HANDLING
# ═══════════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ex: 'Roy canteen phone' or 'How to calculate CGPA?'"):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            try:
                # ════════════════════════════════════════════════════════
                # SEND REQUEST TO BACKEND
                # ════════════════════════════════════════════════════════
                # This calls chat() in chat.py which:
                # 1. Classifies intent (classifier.py)
                # 2. Gets raw data (database/RAG)
                # 3. Formats response (response.py)
                # 4. Returns JSON
                # ════════════════════════════════════════════════════════
                response = requests.post(
                    API_URL, 
                    json={"text": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract answer
                    bot_reply = data.get("answer", "Sorry, I couldn't generate a response.")
                    
                    # Add metadata footer (optional)
                    intent = data.get("intent", "unknown")
                    confidence = data.get("confidence", 0)
                    used_fallback = data.get("used_fallback", False)
                    
                    # Show metadata in small text
                    metadata = f"\n\n---\n"
                    metadata += f"*Intent: `{intent}` | "
                    metadata += f"Confidence: `{confidence:.2f}` | "
                    metadata += f"AI Fallback: `{'Yes' if used_fallback else 'No'}`*"
                    
                    # Display answer
                    full_response = bot_reply + metadata
                    message_placeholder.markdown(full_response)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                
                else:
                    error_msg = f"Server Error {response.status_code}"
                    message_placeholder.error(error_msg)
            
            except requests.exceptions.ConnectionError:
                error_msg = (
                    "**Connection Refused!**\n\n"
                    "Make sure the backend is running:\n"
                    "```bash\n"
                    "uvicorn main:app --reload\n"
                    "```"
                )
                message_placeholder.error(error_msg)
            
            except requests.exceptions.Timeout:
                error_msg = "**Request Timeout!** The server took too long to respond."
                message_placeholder.error(error_msg)
            
            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                message_placeholder.error(error_msg)