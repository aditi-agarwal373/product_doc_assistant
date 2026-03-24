# 📘 Product Documentation Assistant

A RAG-powered web app that lets you upload any product documentation PDF and ask natural language questions — getting precise, cited answers pulled directly from the document.

**Live demo use case:** Upload a SaaS product manual, API reference, or changelog and ask anything.

---

## 🧠 How It Works

```
User uploads PDF
      ↓
PyMuPDF reads all pages
      ↓
LangChain splits into 700-char overlapping chunks
      ↓
OpenAI embeds each chunk → stored in FAISS index
      ↓
User asks a question
      ↓
Question embedded → top 4 similar chunks retrieved from FAISS
      ↓
GPT-3.5 answers using ONLY those chunks
      ↓
Answer + page number citations shown in UI
```

---

## 🗂️ Project Structure

```
product-doc-assistant/
├── backend/
│   ├── main.py           ← FastAPI server (2 endpoints: /upload, /ask)
│   ├── rag_engine.py     ← Full RAG pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html        ← Complete web UI (HTML + CSS + JS)
├── .env                  ← Your OpenAI API key (never commit this!)
└── README.md
```

---

## ⚙️ Local Setup

### 1. Create project + install dependencies
```bash
mkdir product-doc-assistant && cd product-doc-assistant
mkdir backend frontend
# Copy all files into their respective folders

cd backend
python -m venv venv

# Activate:
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Create a `.env` file in the `backend/` folder:
```
OPENAI_API_KEY=sk-your-key-here
```
Get your key: https://platform.openai.com/api-keys

### 3. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4. Open the frontend
Simply open `frontend/index.html` in your browser.
Or run a quick local server:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

### 5. Test it
1. Click **"Click to upload"** → choose a PDF
2. Click **"Process Document"** → wait ~10-20 seconds
3. Ask any question in the chat box
4. See the answer + page citations!

---

## 🚀 Deploy to Railway (Free Public URL)

Railway gives you a live HTTPS URL for both backend and frontend.

### Step 1 — Push to GitHub
```bash
git init
git add .
# IMPORTANT: create a .gitignore first (see below)
git commit -m "Initial commit: Product Doc Assistant"
git remote add origin https://github.com/YOUR_USERNAME/product-doc-assistant.git
git push -u origin main
```

Create `.gitignore`:
```
.env
__pycache__/
*.pyc
venv/
.DS_Store
```

### Step 2 — Deploy Backend on Railway
1. Go to https://railway.app → Sign up with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repo
4. Railway will detect Python → set the start command:
   ```
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
5. Go to **Variables** tab → add:
   ```
   OPENAI_API_KEY = sk-your-key-here
   ```
6. Click **Deploy** → copy your Railway URL (e.g. `https://your-app.railway.app`)

### Step 3 — Update Frontend API URL
In `frontend/index.html`, find this line:
```javascript
const API_BASE = "http://localhost:8000";
```
Change it to your Railway URL:
```javascript
const API_BASE = "https://your-app.railway.app";
```

### Step 4 — Deploy Frontend on Netlify (Free)
1. Go to https://netlify.com → Sign up
2. Drag & drop your `frontend/` folder onto the Netlify dashboard
3. You get a live URL instantly (e.g. `https://your-app.netlify.app`)

---

## 💰 Cost Estimate

| Action | Cost |
|---|---|
| Embed a 50-page PDF | ~$0.003 |
| Each question asked | ~$0.005 |
| 100 questions/month | ~$0.50 |

---

## 📝 CV Bullet (copy this)

> Built and deployed a full-stack RAG-powered Product Documentation Assistant using FastAPI, LangChain, FAISS, and OpenAI GPT-3.5 — featuring a custom HTML/JS frontend, PDF upload, semantic chunk retrieval, and page-level citation display. Deployed live on Railway + Netlify.

---

## 🔧 Interview Talking Points

When asked about this project, you can speak to every layer:

**Product thinking:**
- Pain point: documentation is dense and hard to search
- Solution: conversational interface over any PDF
- Success metric: answer relevance score, time-to-answer vs manual search

**Technical depth:**
- Why RAG vs fine-tuning? RAG works on any doc without retraining
- Chunk size decision: 700 chars balances context richness vs retrieval precision
- Why FAISS? Free, local, no external dependencies — good for prototypes
- Temperature=0: deterministic answers matter for factual Q&A

**What I'd improve:**
- Add conversation memory for follow-up questions
- Persist FAISS index to disk so it survives server restarts
- Add a feedback button (👍👎) to log answer quality
- Support multiple PDFs simultaneously
- Add an admin panel showing most-asked questions

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| FastAPI | Python web framework, REST API |
| Uvicorn | ASGI server to run FastAPI |
| LangChain | RAG orchestration |
| OpenAI text-embedding-3-small | Semantic embeddings |
| FAISS | Vector similarity search |
| GPT-3.5-turbo | Answer generation |
| PyMuPDF | PDF text extraction |
| Vanilla HTML/CSS/JS | Frontend UI |
| Railway | Backend deployment |
| Netlify | Frontend deployment |
