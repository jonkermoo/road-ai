import os
from dotenv import load_dotenv
load_dotenv()

print("=== Checking Supabase Configuration ===")
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
print(f"SUPABASE_SERVICE_ROLE_KEY: {os.getenv('SUPABASE_SERVICE_ROLE_KEY')[:20]}...")
print(f"SUPABASE_BUCKET: {os.getenv('SUPABASE_BUCKET', 'snaps')}")
print()

print("=== Testing connection to Supabase ===")
import requests
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Test 1: Check if events table exists
print("\n1. Checking if 'events' table exists...")
url = f"{SUPABASE_URL}/rest/v1/events?limit=1"
headers = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
}
try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print("   SUCCESS: Table exists and is accessible")
        print(f"   Response: {r.text[:200]}")
    else:
        print(f"   ERROR: {r.text}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Try to insert an event
print("\n2. Attempting to insert test event...")
from supabase_io import insert_event
try:
    insert_event(evt_type="test", img_url=None)
    print("   SUCCESS: Event inserted")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Query recent events
print("\n3. Querying recent events...")
url = f"{SUPABASE_URL}/rest/v1/events?order=ts.desc&limit=5"
try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        import json
        events = json.loads(r.text)
        print(f"   Found {len(events)} recent events:")
        for evt in events:
            print(f"     - {evt.get('ts')}: {evt.get('type')} (img: {evt.get('img_url', 'None')[:50] if evt.get('img_url') else 'None'})")
    else:
        print(f"   ERROR: {r.text}")
except Exception as e:
    print(f"   ERROR: {e}")
