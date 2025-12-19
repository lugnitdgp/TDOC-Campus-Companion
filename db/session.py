# ================================ DAY - 1 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                   DATABASE SESSION CONFIGURATION                         ║
║              Connection Management & Table Initialization                ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This file sets up the DATABASE CONNECTION for the Campus Companion system.
It creates the database engine, manages sessions, and initializes tables.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  [1] MODELS (db/models.py)                                          │
│      • Defines table structure                                     │
│       ↓                                                             │
│  [2] THIS FILE (db/session.py) ← Database Engine                   │
│      • Creates connection to SQLite                                │
│      • Provides SessionLocal for database queries                  │
│      • init_db() creates actual tables                             │
│       ↓                                                             │
│  [3] API ROUTES (api/routers/chat.py)                               │
│      • Uses SessionLocal() to query data                           │
│      • Example: session = SessionLocal()                           │
│       ↓                                                             │
│  [4] DATABASE FILE (campus_companion.db)                            │
│      • Physical SQLite file storing all data                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────
1. Creates database engine (connection to SQLite file)
2. Provides SessionLocal factory for creating database sessions
3. init_db() function creates all tables from models.py
4. Configures SQLite for use with FastAPI (thread safety)

💡 KEY CONCEPTS:
─────────────────────────────────────────────────────────────────────────
Engine:
  • Low-level database connection
  • Manages connection pooling
  • Translates Python → SQL

Session:
  • Workspace for database operations
  • Tracks changes (inserts, updates, deletes)
  • Commits/rolls back transactions
  • Use pattern: create → query → commit → close

SessionLocal:
  • Factory that creates new sessions
  • Each request should get its own session
  • Call SessionLocal() to create a new session

📊 DATABASE WORKFLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Initialize Database (one-time)
    python3 app.py
    ↓
    init_db() called
    ↓
    Creates campus_companion.db with all tables

Step 2: Query Data (in API routes)
    session = SessionLocal()  # Create session
    ↓
    faculty = session.query(Faculty).all()  # Query data
    ↓
    session.close()  # Clean up

Step 3: Insert Data
    session = SessionLocal()
    ↓
    new_faculty = Faculty(name="Dr. John", ...)
    session.add(new_faculty)
    ↓
    session.commit()  # Save to database
    ↓
    session.close()

🔧 CONFIGURATION DETAILS:
─────────────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./campus_companion.db"
  • sqlite:// = Use SQLite database
  • /// = Absolute path follows
  • ./ = Current directory
  • campus_companion.db = Database filename

engine = create_engine(...)
  • check_same_thread=False: Allow multiple threads (needed for FastAPI)
  • echo=False: Don't print SQL queries (set True for debugging)

SessionLocal = sessionmaker(...)
  • autocommit=False: Manual transaction control (safer)
  • autoflush=False: Manual flush control
  • bind=engine: Connect to our SQLite engine

💻 USAGE EXAMPLES:
─────────────────────────────────────────────────────────────────────────
Example 1: Initialize database (first time)
    from db.session import init_db
    init_db()  # Creates all tables

Example 2: Query in API route
    from db.session import SessionLocal
    from db.models import Canteen
    
    def get_canteen_info(name: str):
        session = SessionLocal()
        try:
            canteen = session.query(Canteen).filter(
                Canteen.name.ilike(f"%{name}%")
            ).first()
            return canteen
        finally:
            session.close()  # Always close!

Example 3: Insert data
    session = SessionLocal()
    new_warden = Warden(
        name="Mr. Smith",
        hall="Hall 12",
        phone="+91-9876543210"
    )
    session.add(new_warden)
    session.commit()
    session.close()

⚠️ IMPORTANT BEST PRACTICES:
─────────────────────────────────────────────────────────────────────────
• Always close sessions (use try/finally or context manager)
• Each API request should have its own session
• Don't share sessions between requests
• Call commit() to save changes
• Call rollback() if errors occur

📝 NOTES:
─────────────────────────────────────────────────────────────────────────
• SQLite is simple but not for large-scale production
• For production, switch to PostgreSQL/MySQL
• Database file created automatically on first init_db() call
• Safe to call init_db() multiple times (won't duplicate tables)
"""

from sqlmodel import SQLModel, create_engine
from sqlalchemy.orm import sessionmaker

# ═══════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION STRING
# ═══════════════════════════════════════════════════════════════════════

DATABASE_URL= "sqlite:///./campus_companion.db"

# ═══════════════════════════════════════════════════════════════════════
# CREATE DATABASE ENGINE
# ═══════════════════════════════════════════════════════════════════════

engine= create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)



# ═══════════════════════════════════════════════════════════════════════
# CREATE SESSION FACTORY
# ═══════════════════════════════════════════════════════════════════════

SessionLocal= sessionmaker (autocommit=False, autoflush=False, bind=engine)

# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE DATABASE TABLES FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def init_db():
    from . import models


    SQLModel.metadata.create_all(bind=engine)

    print("Database initialised successfully!")
    print("Database location: {DATABASE_URL}")
    print("All tables created from db/models.py")
