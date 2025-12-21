"""
Database Testing & Seeding Script
==================================
This script populates the database with sample data and runs test queries.
Use this to verify that database setup is working correctly.

Usage:
    python3 testdb.py
"""

from db.session import init_db, SessionLocal  # Database initialization and session
from db import models  # All table models
from sqlmodel import select  # For building SQL queries
from datetime import date  # For today's date


def seed_data():
    """
    Seed Sample Data into Database
    
    Creates dummy records for testing:
    - 2 Faculty members (CSE and ECE departments)
    - 1 Canteen
    - 1 Building with 1 Room
    - 1 Mess Menu for today
    """
    # Initialize database (creates tables if they don't exist)
    init_db()
    
    # Create a new database session
    session = SessionLocal()
    
    try:
        # ===================================================================
        # CREATE FACULTY MEMBERS
        # ===================================================================
        # Create first faculty: Anita Pal from Mathematics department
        f1 = models.Faculty(
            name="Anita Pal",
            department="Mathematics",  # ✅ FIXED: Changed from dept to department
            office_location="LG-15",    # ✅ FIXED: Changed from office to office_location
            email="apal.maths@nitdgp.ac.in",
            phone="+91-9434788069"
        )
        
        # Create second faculty: Shibendu Shekhar Roy from Mechanical department
        f2 = models.Faculty(
            name="Animesh Dutta",
            department="Computer Science and Engineering",  # ✅ FIXED: Changed from dept to department
            office_location="CS-201",             # ✅ FIXED: Changed from office to office_location
            email="adutta.cse@nitdgp.ac.in",
            phone="+91-9434788180"
        )
        
        # Add both faculty members to session (queued for insertion)
        session.add_all([f1, f2])
        
        # ===================================================================
        # CREATE CANTEEN
        # ===================================================================
        c1 = models.Canteen(
            name="Roy Canteen",
            phone="+91-8012345678",
            email="roy@campus.edu",      # ✅ Add email field
            location="Near North Gate"
        )

        c2 = models.Canteen(
            name="Wonders",
            phone="+91-9531662458",
            email="wonders@campus.edu",  # ✅ Add email field
            location="Near MAB"
        )
        session.add_all([c1, c2])  # Queue for insertion
        
        # ===================================================================
        # CREATE BUILDING
        # ===================================================================
        b1 = models.Building(
            name="New Academic Building",
            code="NAB",  # Short code for building
            address="Main Campus",
            lat=28.5449,   # ✅ Add latitude if required
            lng=77.1925    # ✅ Add longitude if required
        )
        session.add(b1)  # Queue for insertion
        
        # IMPORTANT: Commit to save building and get its auto-generated ID
        session.commit()
        
        # Refresh b1 to load the ID that database assigned
        session.refresh(b1)
        # Now b1.id contains the actual database ID (e.g., 1)
        
        # ===================================================================
        # CREATE ROOM (needs building_id from above)
        # ===================================================================
        r1 = models.Room(
            room_no="NAB-201",
            building="New Academic Building",  # ✅ Building name (not ID)
            floor="2nd Floor",                 # ✅ Add floor
            map_link="https://maps.example/nab201"
        )

        r2 = models.Room(
            room_no="NAB-403",
            building="New Academic Building",  # ✅ Building name
            floor="4th Floor",                 # ✅ Add floor
            map_link="https://maps.example/nab403"
        )
        session.add_all([r1, r2])  # Queue for insertion
        
        # ===================================================================
        # Wardens
        # ===================================================================
        w1 = models.Warden(
            name="Dr. Suvamoy Changder", 
            hall="Hall 1", 
            phone="9434788094"
        )
        w2 = models.Warden(
            name="Dr. Sapana Ranwa",
            hall="Hall 13",
            phone="9434789034"
        )

        session.add_all([w1, w2])

        # Commit all remaining changes to database
        session.commit()
        
        print("Sample data seeded.")
        
    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close session to free up resources
        session.close()


def run_queries():
    """
    Run Test Queries to Verify Data
    
    Fetches and displays:
    1. All faculty members
    2. All canteens
    3. Rooms in building with code "AB"
    """
    # Create new session for querying
    session = SessionLocal()
    
    try:
        # ===================================================================
        # QUERY 1: Get all faculty members
        # ===================================================================
        print("\n -- Faculty list --")
        
        # Build SELECT query for Faculty table
        stmt = select(models.Faculty)
        
        # Execute query and loop through results
        for f in session.execute(stmt).scalars():
            # ✅ FIXED: Use department and office_location (not dept and office)
            print(f"{f.id}: {f.name} ({f.department}) - {f.phone} - {f.office_location}")
        
        # ===================================================================
        # QUERY 2: Get all canteens
        # ===================================================================
        print("\n  -- Canteens --")
        
        stmt = select(models.Canteen)
        for c in session.execute(stmt).scalars():
            print(f"{c.id}: {c.name} - {c.phone} - {c.location}")
        
        # ===================================================================
        # QUERY 3: Get rooms
        # ===================================================================
        print("\n -- All Rooms --")
        
        stmt = select(models.Room)
        for r in session.execute(stmt).scalars():
            print(f"{r.id}: {r.room_no} ({r.building}, {r.floor}) - {r.map_link}")

        # ===================================================================
        # QUERY 4: Get all wardens
        # ===================================================================
        print("\n -- Warden List --")
        
        stmt = select(models.Warden)
        for w in session.execute(stmt).scalars():
            print(f"{w.id}: {w.name} - {w.hall} - {w.phone}")
                
    except Exception as e:
        print(f" Query error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close session
        session.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Script entry point
    
    Execution flow:
    1. Seed database with sample data
    2. Run test queries to display data
    """
    print("\n" + "="*70)
    print("DATABASE INITIALIZATION")
    print("="*70 + "\n")
    
    seed_data()      # Insert dummy data
    run_queries()    # Query and display data
    
    print("\n" + "="*70)
    print("🎉 DATABASE READY!")
    print("="*70 + "\n")