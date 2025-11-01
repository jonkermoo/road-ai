import os, json, uuid, requests
from datetime import datetime, timezone
from typing import Optional

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "snaps")

_session = requests.Session()

def _public_url(bucket: str, key: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{key}"

def upload_jpeg(jpg_bytes: bytes, key: Optional[str] = None) -> str:
    """Upload JPEG to Supabase Storage. Returns public URL."""
    if key is None:
        key = f"{uuid.uuid4().hex}.jpg"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{key}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "image/jpeg",
    }
    r = _session.post(url, headers=headers, data=jpg_bytes, timeout=15)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text}")
    return _public_url(SUPABASE_BUCKET, key)

def insert_event(evt_type: str, img_url: Optional[str]) -> None:
    """Insert minimal event row into 'events' (id auto, ts default now())."""
    url = f"{SUPABASE_URL}/rest/v1/events"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": evt_type,
        "img_url": img_url,
    }
    r = _session.post(url, headers=headers, data=json.dumps(payload), timeout=15)
    if r.status_code not in (200, 201, 204):
        raise RuntimeError(f"Supabase insert failed: {r.status_code} {r.text}")