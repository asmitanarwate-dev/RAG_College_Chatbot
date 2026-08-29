from pathlib import Path
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from google import genai
import certifi


# ==========================================
# Load environment variables
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
# PDF documents path
# ==========================================

DOCUMENTS_DIR = Path(
    r"C:\Users\user5\OneDrive\Desktop\Downloads\RAG_College_Chatbot_Documents"
)


# ==========================================
# MongoDB configuration
# ==========================================

DATABASE_NAME = "rag_college_chatbot"
COLLECTION_NAME = "documents"


# ==========================================
# MongoDB connection
# ==========================================

client = MongoClient(
    MONGODB_URL,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=15000
)

client.admin.command("ping")

print("MongoDB connection successful.")


db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# ==========================================
# Gemini client
# ==========================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini Embedding API initialized.")


# ==========================================
# Text splitter
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


# ==========================================
# Read PDF files
# ==========================================

pdf_files = list(
    DOCUMENTS_DIR.glob("*.pdf")
)

print(f"\nFound {len(pdf_files)} PDF files.")


documents_to_insert = []


# ==========================================
# Process PDFs
# ==========================================

for pdf_file in pdf_files:

    print(f"\nReading: {pdf_file.name}")

    reader = PdfReader(
        str(pdf_file)
    )

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"


    chunks = text_splitter.split_text(
        full_text
    )

    print(
        f"Pages: {len(reader.pages)}"
    )

    print(
        f"Characters: {len(full_text)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )


    # ======================================
    # Create Gemini embeddings
    # ======================================

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        embedding_response = (
            gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
        )

        embedding = (
            embedding_response
            .embeddings[0]
            .values
        )


        documents_to_insert.append({

            "source": pdf_file.name,

            "chunk_number": chunk_number,

            "text": chunk,

            "embedding": list(embedding)

        })


        print(
            f"Embedding created: chunk {chunk_number}/{len(chunks)}"
        )


# ==========================================
# Store documents in MongoDB
# ==========================================

if documents_to_insert:

    # Remove old documents
    collection.delete_many({})


    # Insert new documents
    result = collection.insert_many(
        documents_to_insert
    )


    print("\n================================")
    print("INGESTION COMPLETED")
    print("================================")

    print(
        f"PDFs processed: {len(pdf_files)}"
    )

    print(
        f"Documents inserted: {len(result.inserted_ids)}"
    )

    print(
        "Gemini embeddings stored in MongoDB Atlas."
    )

else:

    print(
        "No documents found."
    )


# ==========================================
# Close MongoDB connection
# ==========================================

client.close()

print(
    "\nMongoDB connection closed."
)