import os
import re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Campus Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.info("Please add GOOGLE_API_KEY=your_api_key inside .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

PDF_FOLDER = "documents"

# Current Gemini model
MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f7fa;
        border: 1px solid #ddd;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .feature-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f7f9fc;
        border: 1px solid #e1e5ea;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    all_text = ""

    if not os.path.exists(PDF_FOLDER):
        return ""

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
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


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

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


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    query,
    chunks,
    top_k=5
):

    if not chunks:
        return []

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                chunk.lower()
            )
        )

        score = len(
            query_words.intersection(
                chunk_words
            )
        )

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


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are an AI Campus Assistant.

Answer the student's question using ONLY the information
available in the college document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the information is not available, say:
   "I couldn't find this information in the provided college documents."
4. Give simple and student-friendly answers.
5. Use headings and bullet points when useful.
6. For exam-related questions, clearly highlight important points.

DOCUMENT CONTEXT:
--------------------------------------------------
{context}
--------------------------------------------------

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# GENERATE NOTES
# ============================================================

def generate_notes(context):

    prompt = f"""
You are an AI college study assistant.

Create simple and useful study notes using ONLY the
college document content below.

Include:

• Topic headings
• Important definitions
• Key points
• Short explanations
• Exam-important points

Do not add information that is not present
in the document.

DOCUMENT:
--------------------------------------------------
{context}
--------------------------------------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(context):

    prompt = f"""
You are an AI college quiz generator.

Using ONLY the document content below, create:

10 multiple-choice questions.

For each question provide:

Question
A)
B)
C)
D)

Then provide:

Correct Answer

Do not use information outside the document.

DOCUMENT:
--------------------------------------------------
{context}
--------------------------------------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# IMPORTANT TOPICS
# ============================================================

def generate_important_topics(context):

    prompt = f"""
You are an AI college exam preparation assistant.

Using ONLY the college document content below,
identify the most important topics for exam preparation.

Organize them as:

🔥 HIGH PRIORITY
⭐ MEDIUM PRIORITY
📌 OTHER IMPORTANT TOPICS

For each topic give a short reason why it is important.

Do not invent information.

DOCUMENT:
--------------------------------------------------
{context}
--------------------------------------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# STUDY PLAN
# ============================================================

def generate_study_plan(context, days):

    prompt = f"""
You are an AI college study planner.

Create a {days}-day study plan based ONLY on the
subjects and topics found in the college documents.

For each day include:

Day
Topics
Revision
Practice

Make the plan realistic and student-friendly.

DOCUMENT:
--------------------------------------------------
{context}
--------------------------------------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# LOAD DOCUMENT DATA
# ============================================================

with st.spinner("📚 Loading college documents..."):

    document_text = load_documents()

    chunks = create_chunks(
        document_text
    )


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎓 AI Campus Assistant"
    )

    st.caption(
        "Your smart college study companion"
    )

    st.divider()

    st.markdown(
        "### 🚀 Quick Tools"
    )

    tool = st.radio(
        "Choose a feature",
        [
            "💬 Ask Documents",
            "📝 Generate Notes",
            "❓ Generate Quiz",
            "🎯 Important Topics",
            "📚 Study Planner"
        ]
    )
    st.divider()

# 📤 Upload PDF
st.markdown("### 📤 Upload College PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    os.makedirs(PDF_FOLDER, exist_ok=True)

    file_path = os.path.join(
        PDF_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.cache_data.clear()

    st.success(
        f"✅ {uploaded_file.name} uploaded successfully!"
    )

    st.rerun()
    
    st.divider()

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    st.markdown(
        "### 💬 Chat History"
    )

    if st.session_state.chat_history:

        for i, chat in enumerate(
            st.session_state.chat_history,
            start=1
        ):

            st.caption(
                f"{i}. {chat['question']}"
            )

    else:

        st.caption(
            "No questions asked yet."
        )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Campus Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your smart college study companion'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOOL 1 — ASK DOCUMENTS
# ============================================================

if tool == "💬 Ask Documents":

    st.subheader(
        "💬 Ask Your College Documents"
    )

    st.write(
        "Ask questions about subjects, syllabus, exams, credits and more."
    )

    question = st.chat_input(
        "Ask something about your college documents..."
    )

    if question:

        with st.chat_message("user"):

            st.write(question)

        if not chunks:

            with st.chat_message("assistant"):

                st.error(
                    "❌ No documents available."
                )

        else:

            with st.spinner(
                "🔍 Searching documents..."
            ):

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

                with st.spinner(
                    "🤖 Gemini is thinking..."
                ):

                    try:

                        answer = generate_answer(
                            question,
                            context
                        )

                        # ------------------------------------
                        # SHOW ANSWER
                        # ------------------------------------

                        with st.chat_message("assistant"):

                            st.markdown(
                                '<div class="answer-box">',
                                unsafe_allow_html=True
                            )

                            st.markdown(answer)

                            st.markdown(
                                "</div>",
                                unsafe_allow_html=True
                            )

                        # ------------------------------------
                        # SAVE CHAT
                        # IMPORTANT: AFTER ANSWER EXISTS
                        # ------------------------------------

                        st.session_state.chat_history.append(
                            {
                                "question": question,
                                "answer": answer
                            }
                        )

                    except Exception as e:

                        with st.chat_message("assistant"):

                            st.error(
                                f"❌ Gemini Error: {e}"
                            )


# ============================================================
# TOOL 2 — GENERATE NOTES
# ============================================================

elif tool == "📝 Generate Notes":

    st.subheader(
        "📝 Generate Study Notes"
    )

    st.write(
        "Generate simple study notes from your college documents."
    )

    if st.button(
        "📝 Generate Notes",
        use_container_width=True
    ):

        if not chunks:

            st.error(
                "❌ No documents available."
            )

        else:

            context = "\n\n".join(
                chunks[:8]
            )

            with st.spinner(
                "🤖 Generating notes..."
            ):

                try:

                    notes = generate_notes(
                        context
                    )

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(notes)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


# ============================================================
# TOOL 3 — GENERATE QUIZ
# ============================================================

elif tool == "❓ Generate Quiz":

    st.subheader(
        "❓ Generate Quiz"
    )

    st.write(
        "Test your knowledge using questions from your college documents."
    )

    if st.button(
        "❓ Generate Quiz",
        use_container_width=True
    ):

        if not chunks:

            st.error(
                "❌ No documents available."
            )

        else:

            context = "\n\n".join(
                chunks[:8]
            )

            with st.spinner(
                "🤖 Creating quiz..."
            ):

                try:

                    quiz = generate_quiz(
                        context
                    )

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(quiz)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


# ============================================================
# TOOL 4 — IMPORTANT TOPICS
# ============================================================

elif tool == "🎯 Important Topics":

    st.subheader(
        "🎯 Important Exam Topics"
    )

    st.write(
        "Find high-priority topics from your college documents."
    )

    if st.button(
        "🎯 Find Important Topics",
        use_container_width=True
    ):

        if not chunks:

            st.error(
                "❌ No documents available."
            )

        else:

            context = "\n\n".join(
                chunks[:10]
            )

            with st.spinner(
                "🔍 Finding important topics..."
            ):

                try:

                    topics = generate_important_topics(
                        context
                    )

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(topics)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


# ============================================================
# TOOL 5 — STUDY PLANNER
# ============================================================

elif tool == "📚 Study Planner":

    st.subheader(
        "📚 AI Study Planner"
    )

    st.write(
        "Create a personalized study plan from your college documents."
    )

    days = st.slider(
        "How many days do you want to study?",
        min_value=1,
        max_value=30,
        value=7
    )

    if st.button(
        "📚 Create Study Plan",
        use_container_width=True
    ):

        if not chunks:

            st.error(
                "❌ No documents available."
            )

        else:

            context = "\n\n".join(
                chunks[:10]
            )

            with st.spinner(
                "🤖 Creating your study plan..."
            ):

                try:

                    study_plan = generate_study_plan(
                        context,
                        days
                    )

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        study_plan
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

if tool == "💬 Ask Documents":

    st.divider()

    st.markdown(
        "### 💡 Example Questions"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            "📚 What subjects are there?"
        )

        st.write(
            "🎓 How many credits are there?"
        )

    with col2:

        st.write(
            "📖 Explain the syllabus"
        )

        st.write(
            "🎯 What are the important topics?"
        )

    with col3:

        st.write(
            "📅 What is the exam schedule?"
        )

        st.write(
            "📝 Give me important questions"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI Campus Assistant • Powered by Gemini"
)