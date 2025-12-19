# ================================ DAY - 1 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     FALLBACK AI RESPONSE SYSTEM                          ║
║                         Using Mistral-7B AI                              ║
╚══════════════════════════════════════════════════════════════════════════╝

What Problem Does This Solve?
──────────────────────────────────────────────────────────────────────────
Sometimes users ask questions that aren't in our database or documents:
  • "What's the weather like today?"
  • "How do I make friends on campus?"
  • "Tell me a joke"

Instead of saying "I don't know", we use AI to generate helpful responses!


When Is This Used?
──────────────────────────────────────────────────────────────────────────
  User Query
      │
      ├──→ Classifier: What type of question?
      │       │
      │       ├──→ "Roy canteen phone" → db_contact
      │       │    ├─→ Search Database
      │       │    └─→ Found! Return phone number ✓
      │       │
      │       └──→ "What's the weather?" → ai_fallback
      │            ├─→ Search Database ✗ (no weather data)
      │            ├─→ Search RAG ✗ (no weather docs)
      │            └─→ USE THIS MODULE! ← Fallback AI Response


How It Works: The Flow
──────────────────────────────────────────────────────────────────────────
  1. User asks: "What's the best time to visit the library?"
     │
  2. Database: No "best time" data ✗
     │
  3. RAG: No "best time" documents ✗
     │
  4. FALLBACK TRIGGERED! Call fallback_ai_response()
     │
  5. Initialize Mistral-7B AI model
     │
  6. Create messages:
     ├─→ System: "You are a helpful campus assistant..."
     └─→ Human: "What's the best time to visit the library?"
     │
  7. Send to AI model
     │
  8. AI generates natural response:
     "The library is typically less crowded in the morning hours..."
     │
  9. Return response to user ✓


Technical Details
──────────────────────────────────────────────────────────────────────────
Model: Mistral-7B-Instruct-v0.2
  • 7 billion parameters (very smart!)
  • Instruction-tuned (follows directions well)
  • Hosted on HuggingFace (free tier available)

Authentication:
  • Requires HUGGINGFACEHUB_API_TOKEN in .env file
  • Get free token from: https://huggingface.co/settings/tokens

Parameters:
  • max_new_tokens: 200 (response length limit)
  • temperature: 0.7 (creativity level)
    - 0.0 = deterministic (same answer every time)
    - 1.0 = creative (different answers)
    - 0.7 = balanced (recommended)


Usage Example
──────────────────────────────────────────────────────────────────────────
In your chat.py:

    from core.fallback_message import fallback_ai_response
    
    # After checking database and RAG
    if not db_result and not rag_result:
        # No data found, use AI fallback
        response = fallback_ai_response(user_query)
        return {"answer": response, "used_fallback": True}

Example interaction:
    >>> query = "What should I do if I lose my ID card?"
    >>> response = fallback_ai_response(query)
    >>> print(response)
    "If you lose your ID card, you should immediately report it to
     the campus administration office. They can issue a replacement..."


Error Handling
──────────────────────────────────────────────────────────────────────────
If anything goes wrong:
  • API key missing → Friendly error message
  • Network timeout → Friendly error message  
  • API rate limit → Friendly error message

Never crashes! Always returns something helpful.

Author: Campus Companion Team
Model: mistralai/Mistral-7B-Instruct-v0.2
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
load_dotenv()

def fallback_ai_response(query:str) ->str:


# ===========================================================================
# MAIN FUNCTION: FALLBACK AI RESPONSE
# ===========================================================================

   return (
        "I'm **Campus Companion**, designed to assist specifically with campus-related information such as:\n\n"
        "Contact details (faculty, canteen, hostel, administration)\n\n"
        "Building, room, and facility locations\n\n"
        "Academic rules, CGPA policies, and hostel guidelines\n\n"
        "For the best help, please ask something related to your campus. I'll be happy to assist!"
    )
