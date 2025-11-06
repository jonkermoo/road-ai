from dotenv import load_dotenv
load_dotenv()

from supabase_io import insert_event

# Test inserting an event
print("Inserting test event to Supabase...")
insert_event(evt_type="pothole", img_url=None)
print("Event inserted successfully!")
print("Check your Supabase dashboard to verify.")
