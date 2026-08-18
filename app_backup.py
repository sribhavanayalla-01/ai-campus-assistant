import os
import re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=API_KEY)

PDF_FOLDER = "documents"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Campus Assistant",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f7fa;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 AI Campus Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about your college documents</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD PDF DOCUMENTS
# --------------------------------------------------

@st.cache_data
def load_documents():

    all_text = ""

    if not os.path.exists(PDF_FOLDER):
        return ""

    pdf_files = [
        file for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return ""

    for file in pdf_files:

        pdf_path = os.path.join(PDF_FOLDER, file)

        try:
            reader = PdfReader(pdf_path)

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    all_text += "\n" + text

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return all_text


# --------------------------------------------------
# SPLIT DOCUMENT INTO CHUNKS
# --------------------------------------------------

@st.cache_data
def create_chunks(text):

    if not text:
        return []

    words = text.split()

    chunks = []

    chunk_size = 500
    overlap = 100

    start = 0

    while start < len(words):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# SIMPLE RETRIEVAL
# --------------------------------------------------

def retrieve_relevant_chunks(query, chunks, top_k=5):

    if not chunks:
        return []

    query_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
    )

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                chunk.lower()
            )
        )

        score = len(query_words.intersection(chunk_words))

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]


# --------------------------------------------------
# GEMINI ANSWER
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are an AI Campus Assistant.

Answer the student's question using ONLY the information
provided in the document context below.

If the answer is not available in the context,
clearly say:

"I couldn't find this information in the provided college documents."

Do not invent information.

Give the answer in a simple and student-friendly way.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("📚 Loading college documents..."):

    document_text = load_documents()

    chunks = create_chunks(document_text)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📊 Document Status")

    if document_text:

        st.success("✅ Documents loaded")

        st.write(
            f"📄 Characters: {len(document_text):,}"
        )

        st.write(
            f"🧩 Chunks: {len(chunks)}"
        )

    else:

        st.error("❌ No PDF found")

        st.info(
            "Put your PDF inside the "
            "`documents` folder."
        )

    st.divider()

    st.write("### 💡 Example Questions")

    st.write("• What subjects are there?")

    st.write("• What is the exam schedule?")

    st.write("• Explain the syllabus")

    st.write("• What are the important topics?")

    st.write("• How many credits are there?")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your college documents..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Show user question

    with st.chat_message("user"):
        st.write(question)

    # Check documents

    if not chunks:

        with st.chat_message("assistant"):

            st.error(
                "❌ No documents available. "
                "Please place a PDF inside the documents folder."
            )

    else:

        # Retrieve relevant information

        with st.spinner("🔍 Searching documents..."):

            relevant_chunks = retrieve_relevant_chunks(
                question,
                chunks,
                top_k=5
            )

        if not relevant_chunks:

            with st.chat_message("assistant"):

                st.warning(
                    "I couldn't find relevant information "
                    "in the provided documents."
                )

        else:

            context = "\n\n".join(
                relevant_chunks
            )

            # Generate answer

            with st.spinner("🤖 Gemini is thinking..."):

                try:

                    answer = generate_answer(
                        question,
                        context
                    )

                    with st.chat_message("assistant"):

                        st.markdown(
                            '<div class="answer-box">',
                            unsafe_allow_html=True
                        )

                        st.write(answer)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                except Exception as e:

                    with st.chat_message("assistant"):

                        st.error(
                            f"❌ Gemini Error: {e}"
                        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🎓 AI Campus Assistant • Powered by Gemini"
)