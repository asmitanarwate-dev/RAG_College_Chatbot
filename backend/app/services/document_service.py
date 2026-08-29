from pathlib import Path
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from google import genai
import certifi


# ==========================================
# Environment
# ==========================================

load_dotenv(
    r"backend\app\core\.env",
    override=True
)

MONGODB_URL = os.getenv("MONGODB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL is not configured in .env")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in .env")


# ==========================================
# MongoDB
# ==========================================

DATABASE_NAME = "rag_college_chatbot"
COLLECTION_NAME = "documents"

client = MongoClient(
    MONGODB_URL,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=15000
)

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# ==========================================
# Gemini
# ==========================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# Text splitter
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


# ==========================================
# Process uploaded PDF
# ==========================================

def process_pdf(file_path: str, original_filename: str):

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError("PDF file not found.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")


    # --------------------------------------
    # Read PDF
    # --------------------------------------

    reader = PdfReader(str(pdf_path))

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"


    if not full_text.strip():
        raise ValueError(
            "No readable text found in the PDF."
        )


    # --------------------------------------
    # Create chunks
    # --------------------------------------

    chunks = text_splitter.split_text(
        full_text
    )


    documents = []


    # --------------------------------------
    # Create embeddings
    # --------------------------------------

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )

        embedding = (
            response
            .embeddings[0]
            .values
        )


        documents.append({

            "source": original_filename,

            "chunk_number": chunk_number,

            "text": chunk,

            "embedding": list(embedding)

        })


    # --------------------------------------
    # Insert into MongoDB
    # --------------------------------------

    if documents:

        result = collection.insert_many(
            documents
        )

        return {
            "source": original_filename,
            "pages": len(reader.pages),
            "chunks": len(documents),
            "inserted": len(result.inserted_ids)
        }


    return {
        "source": original_filename,
        "pages": len(reader.pages),
        "chunks": 0,
        "inserted": 0
    }