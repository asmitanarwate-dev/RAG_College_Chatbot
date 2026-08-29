import os

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from google import genai


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
    raise ValueError(
        "MONGODB_URL is not configured in .env"
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
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

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# ==========================================
# Gemini client
# ==========================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# Retrieve relevant documents
# ==========================================

def retrieve_context(query, top_k=3):
    """
    Find the most relevant college document chunks
    using Gemini embeddings and cosine similarity.
    """

    # Create embedding for the user's question
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    query_embedding = np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )

    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        return []

    query_embedding = query_embedding / query_norm


    # Get documents from MongoDB
    documents = list(
        collection.find({})
    )

    if not documents:
        return []


    scored_documents = []


    # Calculate cosine similarity
    for document in documents:

        embedding = np.array(
            document["embedding"],
            dtype=np.float32
        )

        document_norm = np.linalg.norm(
            embedding
        )

        if document_norm == 0:
            continue

        embedding = embedding / document_norm

        score = np.dot(
            query_embedding,
            embedding
        )

        scored_documents.append(
            (
                float(score),
                document
            )
        )


    # Highest similarity first
    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )


    # Return top results
    results = []

    for score, document in scored_documents[:top_k]:

        results.append({
            "text": document["text"],
            "source": document["source"],
            "score": score
        })


    return results