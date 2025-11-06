## How to Use Locally

### Step 1: Start the Backend

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 2: Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Step 3: Access from Your Phone

1. **Find your computer's IP address**:

   - Windows: `ipconfig` (look for IPv4 Address)
   - Mac/Linux: `ifconfig` or `ip addr`

2. **On your phone's browser**, navigate to:

   ```
   http://YOUR_COMPUTER_IP:5173
   ```

   Example: `http://192.168.1.100:5173`

3. **Allow GPS permissions** when prompted
