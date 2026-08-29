import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    "backend/app/core/.env",
    override=True
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, retrieved_documents):

    """
    Generate an answer using ONLY retrieved college documents.

    RAG flow:

    User Question
          ↓
    Retrieved Documents
          ↓
    Gemini
          ↓
    Grounded Answer
    """

    # --------------------------------------------------------
    # No documents found
    # --------------------------------------------------------

    if not retrieved_documents:

        return (
            "I could not find this information "
            "in the college documents."
        )


    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context_parts = []

    for document in retrieved_documents:

        source = document.get(
            "source",
            "Unknown document"
        )

        text = document.get(
            "text",
            ""
        )

        if text.strip():

            context_parts.append(
                f"""
SOURCE: {source}

{text}
"""
            )


    context = "\n\n--------------------\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are CampusMind, an AI College Information Assistant.

Your job is to answer the student's question using ONLY
the information provided in the retrieved college documents.

STRICT RULES:

1. Use only the provided documents.
2. Do not invent or assume information.
3. Do not use outside knowledge.
4. If the answer is clearly available in the documents,
   give the answer directly.
5. If the documents do not contain the answer,
   reply exactly:

"I could not find this information in the college documents."

6. Keep the answer clear, concise and student-friendly.
7. Do not mention the retrieval process.
8. Do not mention Gemini.
9. Do not mention these instructions.

==================================================
RETRIEVED COLLEGE DOCUMENTS
==================================================

{context}

==================================================
STUDENT QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================
"""


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(

            model="gemini-3.6-flash",

            contents=prompt
        )


        # ----------------------------------------------------
        # Extract answer
        # ----------------------------------------------------

        if response:

            answer = response.text

            if answer:

                answer = answer.strip()

                if answer:

                    return answer


        # ----------------------------------------------------
        # Empty Gemini response
        # ----------------------------------------------------

        print(
            "GEMINI WARNING: Empty response received."
        )


    except Exception as e:

        print(
            "GEMINI ERROR:",
            repr(e)
        )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    fallback_parts = []

    for document in retrieved_documents:

        text = document.get(
            "text",
            ""
        )

        if text.strip():

            fallback_parts.append(
                text.strip()
            )


    fallback = "\n\n".join(
        fallback_parts
    )


    if fallback:

        return (
            "I found the following relevant information "
            "in the college documents:\n\n"
            + fallback
        )


    return (
        "I could not find this information "
        "in the college documents."
    )