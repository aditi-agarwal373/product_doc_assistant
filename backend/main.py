import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.rag_engine import process_pdf, ask_question

app = FastAPI(
    title="Product Documentation Assistant API",
    description="RAG-powered Q&A over product documentation PDFs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_state = {
    "qa_chain": None,
    "doc_name": None,
    "num_pages": 0,
    "num_chunks": 0,
}

@app.get("/")
async def root():
    return {"status": "Product Documentation Assistant is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        qa_chain, num_chunks, num_pages = process_pdf(tmp_path)
        app_state["qa_chain"] = qa_chain
        app_state["doc_name"] = file.filename
        app_state["num_pages"] = num_pages
        app_state["num_chunks"] = num_chunks

        return {
            "success": True,
            "filename": file.filename,
            "pages": num_pages,
            "chunks": num_chunks,
            "message": f"Successfully processed {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        os.unlink(tmp_path)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: QuestionRequest):
    if not app_state["qa_chain"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

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


@app.get("/status")
async def status():
    return {
        "doc_loaded": app_state["qa_chain"] is not None,
        "doc_name": app_state["doc_name"],
        "num_pages": app_state["num_pages"],
        "num_chunks": app_state["num_chunks"],
    }
