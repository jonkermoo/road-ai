"""
Test script to insert an event with GPS coordinates.
This simulates a detection with location data.
"""
from dotenv import load_dotenv
load_dotenv()

from supabase_io import insert_event

# Example coordinates (New York City)
LAT = 40.7128
LNG = -74.0060

print(f"Inserting test event with GPS coordinates...")
print(f"  Location: Lat {LAT}, Lng {LNG}")
print(f"  Type: pothole")

insert_event(evt_type="pothole", img_url=None, lat=LAT, lng=LNG)

print("Success! Event inserted with GPS data.")
print("Check your Supabase dashboard and the frontend map!")
