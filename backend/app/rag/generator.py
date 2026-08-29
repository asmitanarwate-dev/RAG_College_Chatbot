import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv(
    r"backend\app\core\.env",
    override=True
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_answer(question, retrieved_documents):
    """
    Generate a grounded answer using Gemini.
    If Gemini is temporarily unavailable,
    return the retrieved college information
    instead of crashing the chatbot.
    """

    if not retrieved_documents:
        return (
            "I could not find this information "
            "in the college documents."
        )

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"Source: {document['source']}\n"
            f"{document['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a College Assistant chatbot.

Answer the student's question using ONLY
the information provided in the college documents.

Do not invent information.

If the answer is not available in the documents,
say:

"I could not find this information in the college documents."

College Documents:

{context}

Student Question:

{question}

Give a clear, concise and helpful answer.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()

    except Exception as e:

        print("GEMINI ERROR:", repr(e))

        # Fallback when Gemini quota/service is unavailable
        fallback_parts = []

        for document in retrieved_documents:
            fallback_parts.append(
                document["text"]
            )

        fallback = "\n\n".join(fallback_parts).strip()

        if fallback:
            return (
                "Gemini AI is temporarily unavailable. "
                "Here is the relevant information found "
                "in the college documents:\n\n"
                + fallback
            )

    return (
        "I could not generate an AI answer right now. "
        "Please try again later."
    )