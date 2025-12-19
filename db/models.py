# ================================ DAY - 1 ================================ #
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                       DATABASE MODELS                                    ║
║                  Table Definitions Using SQLModel                        ║
╚══════════════════════════════════════════════════════════════════════════╝

📁 FILE ROLE IN PROJECT:
─────────────────────────────────────────────────────────────────────────
This file defines the DATABASE SCHEMA for the Campus Companion system.
Each class represents a table, and each class attribute represents a column.
SQLModel automatically converts these Python classes into SQL tables.

🔗 HOW IT FITS IN THE ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  [1] MODELS (THIS FILE) ← Database Structure                        │
│      • Defines what data can be stored                              │
│      • Defines relationships between tables                         │
│       ↓                                                             │
│  [2] SESSION (db/session.py)                                        │
│      • Reads these models                                           │
│      • Creates actual tables in SQLite                              │
│       ↓                                                             │
│  [3] API ROUTES (api/routers/chat.py)                               │
│      • Uses these models to query database                          │
│      • Example: session.query(Faculty).filter(...)                  │
│       ↓                                                             │
│  [4] DATABASE FILE (campus_companion.db)                            │
│      • Physical SQLite file with actual data                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🎯 WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────
1. Defines 10 database tables for campus data
2. Specifies columns, types, and constraints for each table
3. Provides type hints for better code completion and error checking
4. Auto-generates SQL CREATE TABLE statements via SQLModel

📊 DATABASE TABLES DEFINED HERE:
─────────────────────────────────────────────────────────────────────────
1. User         - Student/admin accounts
2. Faculty      - Professor information
3. Canteen      - Campus food outlet contacts
4. Warden       - Hostel warden information
5. Building     - Campus building data
6. Room         - Individual room locations
7. MessMenu     - Daily hostel food schedules
8. Document     - Uploaded PDFs and files
9. Notice       - Campus announcements
10. Embedding   - AI vector embeddings (for RAG)

🔑 KEY CONCEPTS:
─────────────────────────────────────────────────────────────────────────
SQLModel = Pydantic + SQLAlchemy
  • Type validation (ensures correct data types)
  • Auto-generates database tables
  • Works with FastAPI seamlessly

Field Types:
  • Optional[int] = Can be None (nullable column)
  • str = Required string (NOT NULL)
  • datetime = Timestamp with date + time
  • date = Date only (no time)
  • dict = JSON object stored in database

Field() Parameters:
  • default=None: Column can be empty
  • primary_key=True: Unique identifier for each row
  • default_factory=datetime.utcnow: Auto-set to current time

💡 EXAMPLE USAGE:
─────────────────────────────────────────────────────────────────────────
Creating a new faculty record:

    from db.models import Faculty
    from db.session import SessionLocal
    
    # Create a new faculty entry
    new_faculty = Faculty(
        name="Dr. John Doe",
        department="Computer Science",
        office_location="AB-301",
        email="john.doe@nitdgp.ac.in",
        phone="+91-9876543210"
    )
    
    # Save to database
    session = SessionLocal()
    session.add(new_faculty)
    session.commit()

Querying faculty:

    from sqlmodel import select
    
    # Find all CS department faculty
    statement = select(Faculty).where(Faculty.department == "Computer Science")
    results = session.exec(statement).all()

🔧 HOW TO MODIFY:
─────────────────────────────────────────────────────────────────────────
Adding a new table:
1. Create a new class inheriting from SQLModel
2. Add table=True parameter
3. Define columns as class attributes
4. Run init_db() to create the table

Example:
    class Student(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        roll_no: str
        name: str
        email: str

Adding a column to existing table:
1. Add attribute to the class
2. Drop old table or use Alembic migrations
3. Re-run init_db()

📝 IMPORTANT NOTES:
─────────────────────────────────────────────────────────────────────────
• These are just DEFINITIONS - no data stored here
• Actual data lives in campus_companion.db file
• Changes here require database re-initialization
• Use Optional[] for nullable columns, plain type for required
"""




# IMPORTING REQUIRED MODULES








# WE WILL START CREATING DATABASE TABLES NOW

# ============================================================================
# USER MODEL
# ============================================================================







# ============================================================================
# FACULTY MODEL
# ============================================================================







# ============================================================================
# CANTEEN MODEL
# ============================================================================

# class Canteen(SQLModel, table=True):
#     """
#     Canteen table for campus food outlets
    
#     Attributes:
#         id: Unique identifier
#         name: Canteen name (e.g., "Roy Canteen")
#         phone: Contact phone number
#         email: Contact email
#         location: Physical location description
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     name: str  # Required - canteen must have a name
#     phone: str  # Contact for orders/queries
#     email: Optional[str] = None  # Contact email
#     location: Optional[str] = None  # Where it's located on campus

# ============================================================================
# WARDEN MODEL
# ============================================================================

# class Warden(SQLModel, table=True):
#     """
#     Warden table for hostel warden information
    
#     Attributes:
#         id: Unique identifier
#         name: Warden's full name
#         hall: Hostel hall name/number
#         phone: Contact phone number
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     name: str  # Required - warden's name
#     hall: Optional[str] = None  # Which hostel they manage
#     phone: Optional[str] = None  # Emergency contact number

# ============================================================================
# BUILDING MODEL
# ============================================================================

# class Building(SQLModel, table=True):
#     """
#     Building table for campus building information
    
#     Attributes:
#         id: Unique identifier
#         name: Full building name (e.g., "Academic Block A")
#         code: Short code (e.g., "AB")
#         address: Physical address description
#         lat: Latitude coordinate for maps
#         lng: Longitude coordinate for maps
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     name: str  # Required - building name
#     code: Optional[str] = None  # Short abbreviation for quick reference
#     address: Optional[str] = None  # Detailed location
#     lat: Optional[float] = None  # GPS latitude
#     lng: Optional[float] = None  # GPS longitude

# ============================================================================
# ROOM MODEL
# ============================================================================

# class Room(SQLModel, table=True):
#     """
#     Room table for individual rooms within buildings
    
#     Attributes:
#         id: Unique identifier
#         room_no: Room number (e.g., "AB-201")
#         building: Building name
#         floor: Floor number
#         map_link: URL to Google Maps or campus map
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     room_no: str  # Unique room identifier
#     building: Optional[str] = None  # Building name
#     floor: Optional[str] = None  # Floor number
#     map_link: str  # Navigation link

# ============================================================================
# DOCUMENT MODEL
# ============================================================================

# class Document(SQLModel, table=True):
#     """
#     Document table for storing uploaded files and PDFs
    
#     Attributes:
#         id: Unique identifier
#         title: Document title/name
#         department: Which department uploaded it
#         storage_path: File path or cloud storage URL
#         extracted_text: Text content extracted from PDF (for search)
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     title: Optional[str] = None  # Document name
#     department: Optional[str] = None  # Source department
#     storage_path: Optional[str] = None  # Where file is stored
#     extracted_text: Optional[str] = None  # OCR/parsed text content

# ============================================================================
# NOTICE MODEL
# ============================================================================

# class Notice(SQLModel, table=True):
#     """
#     Notice table for campus announcements and notifications
    
#     Attributes:
#         id: Unique identifier
#         doc_id: Foreign key linking to Document table
#         title: Notice headline
#         summary: Brief description
#         published_date: When notice was posted
#     """
#     id: Optional[int] = Field(default=None, primary_key=True)
#     doc_id: Optional[int] = None  # Links to Document.id
#     title: Optional[str] = None  # Notice title
#     summary: Optional[str] = None  # Short description
#     published_date: Optional[date] = None  # Publication date

# ============================================================================
# EMBEDDING MODEL
# ============================================================================









