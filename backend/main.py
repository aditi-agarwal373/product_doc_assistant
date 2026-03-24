from backend.rag_engine import process_pdf, ask_question"""
FastAPI Backend for Product Documentation Assistant
----------------------------------------------------
Exposes two endpoints:
  POST /upload  → accepts a PDF, processes it through RAG pipeline
  POST /ask     → accepts a question, returns answer + citations

FastAPI automatically generates API docs at http://localhost:8000/docs
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_engine import process_pdf, ask_question

# ─────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Product Documentation Assistant API",
    description="RAG-powered Q&A over product documentation PDFs",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# CORS MIDDLEWARE
# This allows the frontend (running on a different port or domain)
# to make requests to this backend. Without this, browsers block the requests.
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# IN-MEMORY STATE
# Stores the active QA chain and document metadata.
# NOTE: This resets if the server restarts. For production,
# you'd persist the FAISS index to disk.
# ─────────────────────────────────────────────────────────────
app_state = {
    "qa_chain": None,
    "doc_name": None,
    "num_pages": 0,
    "num_chunks": 0,
}

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# Mounts the frontend folder so index.html is served at "/"
# This means you only need to run ONE server for both frontend + backend
# ─────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: POST /upload
# Accepts a PDF file, runs it through the full RAG pipeline,
# stores the QA chain in app_state for subsequent /ask calls.
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF product documentation file.
    Returns: document stats (pages, chunks processed)
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temporary location
    # We need a real file path (not just bytes) because PyMuPDF requires a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run the full RAG pipeline
        qa_chain, num_chunks, num_pages = process_pdf(tmp_path)

        # Store in app state
        app_state["qa_chain"] = qa_chain
        app_state["doc_name"] = file.filename
        app_state["num_pages"] = num_pages
        app_state["num_chunks"] = num_chunks

        return {
            "success": True,
            "filename": file.filename,
            "pages": num_pages,
            "chunks": num_chunks,
            "message": f"Successfully processed {file.filename} — {num_pages} pages indexed into {num_chunks} searchable chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# Pydantic models define the shape of JSON request/response bodies.
# FastAPI uses these for automatic validation + docs generation.
# ─────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    page: int
    snippet: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    doc_name: str


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: POST /ask
# Accepts a question, retrieves relevant chunks from FAISS,
# sends them to GPT, returns answer + page citations.
# ─────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Ask a question about the uploaded documentation.
    Returns: answer text + list of source pages with snippets
    """
    if not app_state["qa_chain"]:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = ask_question(app_state["qa_chain"], request.question)
        return {
            "answer": answer,
            "sources": sources,
            "doc_name": app_state["doc_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3: GET /status
# Utility endpoint to check if a document has been loaded.
# The frontend calls this on page load to restore state.
# ─────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "doc_loaded": app_state["qa_chain"] is not None,
        "doc_name": app_state["doc_name"],
        "num_pages": app_state["num_pages"],
        "num_chunks": app_state["num_chunks"],
    }
