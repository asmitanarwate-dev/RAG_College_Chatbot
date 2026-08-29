from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil

from backend.app.rag.retriever import retrieve_context
from backend.app.rag.generator import generate_answer
from backend.app.services.document_service import process_pdf


app = FastAPI(
    title="RAG-Based College Chatbot",
    description="AI-powered college information assistant",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Request Model
# ==========================================

class ChatRequest(BaseModel):
    question: str


# ==========================================
# Frontend
# ==========================================

@app.get("/")
def root():
    frontend_path = Path("frontend/index.html")

    if frontend_path.exists():
        return FileResponse(frontend_path)

    return {
        "message": "RAG-Based College Chatbot API is running!",
        "status": "success"
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ==========================================
# Chat API
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # Retrieve relevant documents
        retrieved_documents = retrieve_context(
            question,
            top_k=3
        )

        # Generate grounded answer
        answer = generate_answer(
            question,
            retrieved_documents
        )

        # ======================================
        # Prepare UNIQUE sources
        # ======================================

        sources = []
        seen_sources = set()

        for document in retrieved_documents:

            source = document["source"]

            if source not in seen_sources:

                sources.append({
                    "source": source,
                    "score": document["score"]
                })

                seen_sources.add(source)

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        print("CHAT ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


# ==========================================
# ADMIN - PDF UPLOAD
# ==========================================

@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    # Only PDF allowed
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Upload directory
    upload_dir = Path("uploaded_documents")

    upload_dir.mkdir(
        exist_ok=True
    )

    # Create file path
    file_path = upload_dir / file.filename

    try:

        # Save uploaded PDF
        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # Process PDF
        result = process_pdf(
            str(file_path),
            file.filename
        )

        return {
            "status": "success",
            "message": "PDF uploaded and added to the knowledge base.",
            "document": result
        }

    except Exception as e:

        print("UPLOAD ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        await file.close()