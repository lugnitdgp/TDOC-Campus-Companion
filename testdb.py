"""
Database Testing & Seeding Script
==================================
This script populates the database with sample data and runs test queries.
Use this to verify that database setup is working correctly.

Usage:
    python3 testdb.py
"""

from db.session import init_db, SessionLocal 
from db import models  
from sqlmodel import select 
from datetime import date 


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
        f1 = models.Faculty(
            name="Anita Pal",
            department="Mathematics", 
            office_location="LG-15",  
            email="apal.maths@nitdgp.ac.in",
            phone="+91-9434788069"
        )
        
        f2 = models.Faculty(
            name="Shibendu Shekhar Roy",
            department="Mechanical Engineering",
            office_location="AB-201",             
            email="ssroy.me@nitdgp.ac.in",
            phone="+91-9123456780"
        )
        
        session.add_all([f1, f2])
        
        # ===================================================================
        # CREATE CANTEEN
        # ===================================================================
        c1 = models.Canteen(
            name="Roy Canteen",
            phone="+91-8012345678",
            email="roy@campus.edu",      
            location="Near North Gate"
        )

        c2 = models.Canteen(
            name="Wonders",
            phone="+91-9531662458",
            email="wonders@campus.edu", 
            location="Near MAB"
        )
        session.add_all([c1, c2])  
        
        # ===================================================================
        # CREATE BUILDING
        # ===================================================================
        b1 = models.Building(
            name="New Academic Building",
            code="NAB",  
            address="Main Campus",
            lat=28.5449,   
            lng=77.1925    
        )
        session.add(b1)  
        
        session.commit()
        
        session.refresh(b1)
        
        # ===================================================================
        # CREATE ROOM (needs building_id from above)
        # ===================================================================
        r1 = models.Room(
            room_no="NAB-201",
            building="New Academic Building", 
            floor="2nd Floor",                 
            map_link="https://maps.example/nab201"
        )

        r2 = models.Room(
            room_no="NAB-403",
            building="New Academic Building",  
            floor="4th Floor",               
            map_link="https://maps.example/nab403"
        )
        session.add_all([r1, r2]) 
        
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

        session.commit()
        
        print("Sample data seeded.")
        
    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        session.close()


def run_queries():
    """
    Run Test Queries to Verify Data
    
    Fetches and displays:
    1. All faculty members
    2. All canteens
    3. Rooms in building with code "AB"
    """
    session = SessionLocal()
    
    try:
        # ===================================================================
        # QUERY 1: Get all faculty members
        # ===================================================================
        print("\n -- Faculty list --")
        

        stmt = select(models.Faculty)
        
        for f in session.execute(stmt).scalars():
            print(f"{f.id}: {f.name} ({f.department}) - {f.phone} - {f.office_location}")
        
        # ===================================================================
        # QUERY 2: Get all canteens
        # ===================================================================
        print("\n -- Canteens --")
        
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
        print(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
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
    
    seed_data()     
    run_queries()   
    
    print("\n" + "="*70)
    print(" DATABASE READY!")
    print("="*70 + "\n")