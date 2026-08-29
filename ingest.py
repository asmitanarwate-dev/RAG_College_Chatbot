from pathlib import Path
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import certifi


# Load MongoDB URL
load_dotenv(r"backend\app\core\.env", override=True)

MONGODB_URL = os.getenv("MONGODB_URL")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL not found in .env")


# PDF folder
DOCUMENTS_DIR = Path(
    r"C:\Users\user5\OneDrive\Desktop\Downloads\RAG_College_Chatbot_Documents"
)


# MongoDB
DATABASE_NAME = "rag_college_chatbot"
COLLECTION_NAME = "documents"


print("Connecting to MongoDB...")

client = MongoClient(
    MONGODB_URL,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=15000
)

client.admin.command("ping")

print("MongoDB connection successful.")


db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# Embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")
print(
    "Embedding dimension:",
    model.get_sentence_embedding_dimension()
)


# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


# Find PDFs
pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF files.")


documents_to_insert = []


# Process PDFs
for pdf_file in pdf_files:

    print(f"\nReading: {pdf_file.name}")

    reader = PdfReader(str(pdf_file))

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"


    chunks = text_splitter.split_text(full_text)

    print(f"Pages: {len(reader.pages)}")
    print(f"Characters: {len(full_text)}")
    print(f"Chunks created: {len(chunks)}")


    print("Creating embeddings...")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )


    for chunk_number, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1
    ):

        documents_to_insert.append({
            "source": pdf_file.name,
            "chunk_number": chunk_number,
            "text": chunk,
            "embedding": embedding.tolist()
        })


# Store in MongoDB
if documents_to_insert:

    print("\nRemoving old documents...")

    collection.delete_many({})

    print("Inserting documents into MongoDB...")

    result = collection.insert_many(documents_to_insert)

    print("\n================================")
    print("RAG INGESTION COMPLETED")
    print("================================")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Documents inserted: {len(result.inserted_ids)}")
    print("Embeddings stored successfully.")


else:

    print("No PDF documents found.")


client.close()

print("MongoDB connection closed.")
