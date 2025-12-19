"""
Database Testing & Seeding Script
==================================
This script populates the database with sample data and runs test queries.
Use this to verify that database setup is working correctly.

Usage:
    python3 testdb.py
"""

# ============================================================================
# IMPORTS
# ============================================================================






# ==========================================================================
# DEFINE DATABASE SEEDING FUNCTION
# ==========================================================================






        # ===================================================================
        # CREATE FACULTY MEMBERS
        # ===================================================================
        # Create first faculty: Anita Pal from Mathematics department







        # f2 = models.Faculty(
        #     name="Shibendu Shekhar Roy",
        #     department="Mechanical Engineering",  
        #     office_location="AB-201",             
        #     email="ssroy.me@nitdgp.ac.in",
        #     phone="+91-9123456780"
        # )


        # ===================================================================
        # CREATE CANTEEN
        # ===================================================================





        # c2 = models.Canteen(
        #     name="Wonders",
        #     phone="+91-9531662458",
        #     email="wonders@campus.edu",
        #     location="Near MAB"
        # )


        # ===================================================================
        # CREATE BUILDING
        # ===================================================================

        # b1 = models.Building(
        #     name="New Academic Building",
        #     code="NAB",  # Short code for building
        #     address="Main Campus",
        #     lat=28.5449,   # ✅ Add latitude if required
        #     lng=77.1925    # ✅ Add longitude if required
        # )

        # ===================================================================
        # CREATE ROOM (needs building_id from above)
        # ===================================================================
        # r1 = models.Room(
        #     room_no="NAB-201",
        #     building="New Academic Building",  
        #     floor="2nd Floor",                 
        #     map_link="https://maps.example/nab201"
        # )

        # r2 = models.Room(
        #     room_no="NAB-403",
        #     building="New Academic Building",  
        #     floor="4th Floor",                
        #     map_link="https://maps.example/nab403"
        # )
        # session.add_all([r1, r2])  
        
        # ===================================================================
        # Wardens
        # ===================================================================
        # w1 = models.Warden(
        #     name="Dr. Suvamoy Changder", 
        #     hall="Hall 1", 
        #     phone="9434788094"
        # )
        # w2 = models.Warden(
        #     name="Dr. Sapana Ranwa",
        #     hall="Hall 13",
        #     phone="9434789034"
        # )

        # session.add_all([w1, w2])

        # session.commit()











# ==========================================================================
# DEFINE TEST QUERY FUNCTION
# ==========================================================================






        # ===================================================================
        # QUERY 2: Get all canteens
        # ===================================================================
        # print("\n🍽️  -- Canteens --")
        
        # stmt = select(models.Canteen)
        # for c in session.execute(stmt).scalars():
        #     print(f"{c.id}: {c.name} - {c.phone} - {c.location}")
        
        # ===================================================================
        # QUERY 3: Get rooms
        # ===================================================================
        # print("\n🏢 -- All Rooms --")
        
        # stmt = select(models.Room)
        # for r in session.execute(stmt).scalars():
        #     print(f"{r.id}: {r.room_no} ({r.building}, {r.floor}) - {r.map_link}")

        # ===================================================================
        # QUERY 4: Get all wardens
        # ===================================================================
        # print("\n🏠 -- Warden List --")
        
        # stmt = select(models.Warden)
        # for w in session.execute(stmt).scalars():
        #     print(f"{w.id}: {w.name} - {w.hall} - {w.phone}")




# ============================================================================
# MAIN EXECUTION
# ============================================================================
