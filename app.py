#app.py
import streamlit as st
import uuid
import os
import io
import chromadb
from chromadb.config import Settings
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple, Optional
from agents.summarizer import build_medical_summarizer, SummaryType
from agents.note_agent import build_note_agent, NoteType
from agents.quiz_agent import MCQAgent, QuizMode
from utils.clinical_validator import ClinicalValidator
from datetime import datetime
from functools import partial
import time
from streamlit.components.v1 import html
from typing import Set
from agents.slide_agent import SlideAgent, SlideStyle
from io import BytesIO
# Add to existing imports
from utils.document_processor import DocumentProcessor
from langchain.schema import Document
from helpers import neural_chat_response, display_quiz, detect_task, extract_metadata, format_discharge_summary, show_flashcards, save_chat_history, load_chat_history, show_quiz
from rag_engine import build_vectorstore, ask_with_rag
from agents.flashcards import FlashcardAgent
import json

# Add this to your session state initialization
if 'generated_note' not in st.session_state:
    st.session_state.generated_note = None
if 'pending_task' not in st.session_state:
    st.session_state.pending_task = None
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None
if 'uploaded_files' not in st.session_state:  # NEW: Track uploaded files
    st.session_state.uploaded_files = []
if 'active_file_id' not in st.session_state:
    st.session_state.active_file_id = None
if 'generated_flashcards' not in st.session_state:
    st.session_state.generated_flashcards = []
if 'current_flashcard' not in st.session_state:
    st.session_state.current_flashcard = 0
if 'show_flashcard_answer' not in st.session_state:
    st.session_state.show_flashcard_answer = False
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())[:8]
if 'generated_quiz' not in st.session_state:
    st.session_state.generated_quiz = []
if 'current_quiz_index' not in st.session_state:
    st.session_state.current_quiz_index = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'answered_quiz' not in st.session_state:
    st.session_state.answered_quiz = set()


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history(st.session_state.run_id)
session_id = st.session_state.run_id


# Custom CSS for improved styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stRadio > div {
        display: flex;
        gap: 10px;
    }
    .stRadio [role="radiogroup"] {
        gap: 15px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Add with your other CSS
st.markdown("""
<style>
    .flashcard {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    .flashcard:hover {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


class FileStorageManager:
    def __init__(self):
        try:
            # Initialize ChromaDB with new client configuration
            self.client = chromadb.PersistentClient(
                path=".chromadb"
            )
            
            # Initialize collections
            self._initialize_collections()
            
            os.makedirs("uploads", exist_ok=True)
            self.processor = DocumentProcessor()
            
        except Exception as e:
            st.error(f"Failed to initialize storage: {str(e)}")
            # Fallback to ephemeral client if persistent fails
            self._initialize_fallback()
    
    def _initialize_collections(self):
        """Initialize collections with proper error handling"""
        try:
            self.files_collection = self.client.get_or_create_collection(
                name="uploaded_files",
                metadata={"hnsw:space": "cosine"}
            )
            self.content_collection = self.client.get_or_create_collection(
                name="file_contents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.warning(f"Collection initialization failed: {str(e)}")
            try:
                # Try creating fresh collections if get_or_create fails
                self.files_collection = self.client.create_collection(
                    name="uploaded_files",
                    metadata={"hnsw:space": "cosine"}
                )
                self.content_collection = self.client.create_collection(
                    name="file_contents",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                st.error(f"Failed to create collections: {str(e)}")
                raise RuntimeError("Could not initialize collections")
    
    def _initialize_fallback(self):
        """Fallback initialization when persistent storage fails"""
        try:
            st.warning("Using ephemeral client (data will not persist after session)")
            self.client = chromadb.EphemeralClient()
            self.files_collection = self.client.create_collection("uploaded_files")
            self.content_collection = self.client.create_collection("file_contents")
        except Exception as e:
            st.error(f"Critical storage initialization failed: {str(e)}")
            raise RuntimeError("Could not initialize any storage system")

    def _generate_file_id(self, file_bytes: bytes) -> str:
        """Generate a unique ID based on file content"""
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        return f"file_{file_hash}"

    def store_file(self, file_bytes: bytes, filename: str, metadata: dict, session_id: str) -> dict:
        try:
            file_id = self._generate_file_id(file_bytes)
            file_ext = Path(filename).suffix.lower()
            storage_path = os.path.join("uploads", f"{uuid.uuid4()}{file_ext}")

            with open(storage_path, "wb") as f:
                f.write(file_bytes)

            content = ""
            doc_type = "unknown"

            try:
                if file_ext == '.pdf':
                    file_like = io.BytesIO(file_bytes)
                    pages = self.processor.process_pdf(file_like)
                    content = "\n".join(p.page_content for p in pages)
                    doc_type = pages[0].metadata.get("doc_type", "unknown") if pages else "unknown"

                elif file_ext in ('.txt', '.md'):
                    with open(os.path.join("session_data", filename), "r", encoding="utf-8") as f:
                        content = f.read()
                    doc_type = "text"

                elif file_ext in ('.jpg', '.jpeg', '.png'):
                    from helpers import extract_text_from_image
                    from PIL import Image
                    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    content = extract_text_from_image(image)
                    doc_type = "handwritten"


            except Exception as e:
                st.error(f"File processing error: {str(e)}")

            # Store metadata (basic preview)
            self.files_collection.add(
                documents=[content[:500]],
                metadatas=[{
                    **metadata,
                    "session_id": session_id,
                    "file_path": storage_path,
                    "original_name": filename,
                    "stored_at": datetime.now().isoformat(),
                    "doc_type": doc_type
                }],
                ids=[file_id]
            )

            # Store full content with chunks
            if content:
                chunk_size = 1000
                chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
                chunk_ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]

                self.content_collection.add(
                    documents=chunks,
                    metadatas=[{
                        "file_id": file_id,
                        "session_id": session_id,
                        "chunk_index": i,
                        "content_type": "chunk"
                    } for i in range(len(chunks))],
                    ids=chunk_ids
                )

            return {
                "file_id": file_id,
                "storage_path": storage_path,
                "original_name": filename,
                "content_length": len(content)
            }

        except Exception as e:
            st.error(f"Failed to store file: {str(e)}")
            raise

    def query_content(self, query: str, file_id: str, n_results: int = 3) -> List[Tuple[str, float]]:
        """Query the stored content and return relevant chunks with scores"""
        try:
            results = self.content_collection.query(
                query_texts=[query],
                where={"file_id": file_id},
                n_results=n_results
            )
            return list(zip(results['documents'][0], results['distances'][0]))
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return []

    def get_file_content(self, file_id: str) -> Optional[str]:
        """Get the full text content of a file"""
        try:
            results = self.content_collection.get(
                where={"file_id": file_id},
                include=["documents"]
            )
            if results and results['documents']:
                return "\n".join(results['documents'])
            return None
        except Exception as e:
            st.error(f"Error retrieving file content: {str(e)}")
            return None
        
if 'file_storage' not in st.session_state:
    st.session_state.file_storage = FileStorageManager()


# Cache configuration
@st.cache_resource
def get_processor():
    return DocumentProcessor()

import copy

def safe_json(obj):
    """Convert obj into something JSON serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')  # be cautious with large files
    return obj

# Create a deepcopy to avoid modifying session state directly
chat_history_cleaned = copy.deepcopy(st.session_state.chat_history)
uploaded_files_cleaned = copy.deepcopy(st.session_state.uploaded_files)

# Sanitize fields in chat history (e.g. remove binary content if present)
for msg in chat_history_cleaned:
    if isinstance(msg.get("content"), (bytes, bytearray)):
        msg["content"] = "<binary content removed>"

# Sanitize uploaded file metadata (remove anything non-serializable)
for file in uploaded_files_cleaned:
    file["uploaded_at"] = safe_json(file.get("uploaded_at"))
    file["path"] = str(file.get("path", ""))
    file["size"] = int(file.get("size", 0))

topic = st.session_state.get("task") or st.session_state.get("pending_task") or "General"

save_data = {
    "id": session_id,
    "chat_history": st.session_state.chat_history,
    "uploaded_files": st.session_state.uploaded_files,
    "active_file_id": st.session_state.active_file_id,
    "topic": topic
}


os.makedirs("session_data", exist_ok=True)
with open(f"session_data/{st.session_state.run_id}.json", "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)



@st.cache_resource
def load_quiz_agent():
    return MCQAgent()




processor              = DocumentProcessor()
validator              = ClinicalValidator()
quiz_agent             = MCQAgent()
slide_agent            = SlideAgent()
research_summarizer    = build_medical_summarizer(SummaryType.RESEARCH)
discharge_summarizer   = build_medical_summarizer(SummaryType.DISCHARGE)
note_agent             = build_note_agent(NoteType.CLINICAL)
flashcard_agent        = FlashcardAgent()



# Document Upload Section
uploaded_file = None
pages = None
topic = None


st.title("⚕️ CellBot: Medical Documentation Assistant")
st.markdown("*Ask me to summarize, quiz, take notes, flashcards or discharge summaries.*")

# ── ChatGPT‑style dispatcher ─────────────────────────────────────

# 1) Chat prompt
user_input = st.chat_input("Ask me to summarize, quiz, notes, slides or discharge summary...")

if user_input:
    task = detect_task(user_input.lower())
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_chat_history(st.session_state.run_id, st.session_state.chat_history)


    # File-based tasks need a document
    if task in ("research", "discharge", "note", "slides", "quiz", "flashcards"):
        st.session_state.pending_task = task
        st.session_state.pending_prompt = user_input
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"🧠 Got it. Please upload the document to '{task}'..."
        })
        save_chat_history(st.session_state.run_id, st.session_state.chat_history)


    # Handle follow-up questions using the active file
    else:
        if st.session_state.active_file_id:
            # Find the active file info
            active_file = next((f for f in st.session_state.uploaded_files 
                              if f['id'] == st.session_state.active_file_id), None)
            file_name = active_file['name'] if active_file else "selected document"
            
            with st.spinner(f"🔍 Searching {file_name}..."):
                # Use the FileStorageManager to query content
                relevant_chunks = st.session_state.file_storage.query_content(
                    user_input,
                    st.session_state.active_file_id
                )
                
                if relevant_chunks:
                    # Format context for the LLM
                    context = "\n\n".join([f"Document excerpt {i+1}:\n{chunk}" 
                                         for i, (chunk, score) in enumerate(relevant_chunks)])
                    
                    # Generate answer using both context and the question
                    prompt = f"""
                    Based on the following document excerpts from '{file_name}':
                    {context}
                    
                    Answer this question: {user_input}
                    
                    Instructions:
                    1. Be concise and accurate
                    2. If the answer isn't in the document, say so
                    3. For medical content, include important details like doses
                    4. Format lists clearly when appropriate
                    
                    Answer:"""
                    
                    answer = neural_chat_response(prompt)
                    answer = f"📄 Based on **{file_name}**:\n\n{answer}"
                else:
                    answer = f"I couldn't find relevant information in {file_name} to answer that question."
        else:
            with st.spinner("💬 Thinking..."):
                answer = neural_chat_response(user_input)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
        save_chat_history(st.session_state.run_id, st.session_state.chat_history)


# 2) File uploader if document task pending
uploaded_file = None
if st.session_state.get("pending_task") in ("research", "discharge", "note", "slides", "quiz", "flashcards"):
    uploaded_file = st.file_uploader(
        f"📁 Upload document for '{st.session_state.pending_task}'",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        key="pending_uploader"
    )

# 3) Once file uploaded, extract and respond
if uploaded_file and st.session_state.get("pending_task"):
    task = st.session_state.pending_task
    prompt = st.session_state.pending_prompt

    # Store the file and get metadata
    file_meta = st.session_state.file_storage.store_file(
        uploaded_file.getvalue(),
        uploaded_file.name,
        {"task": task},
        st.session_state.run_id
    )

    # Track the uploaded file (NEW)
    st.session_state.uploaded_files.append({
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "size": len(uploaded_file.getvalue()),
        "path": file_meta["storage_path"],
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "id": file_meta["file_id"]
    })

    # Store the current file ID for RAG
    st.session_state.current_file_id = file_meta["file_id"]

    # Extract content depending on file type
    raw_text = st.session_state.file_storage.get_file_content(file_meta["file_id"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"⏳ Processing your {task}..."
    })
    save_chat_history(st.session_state.run_id, st.session_state.chat_history)


    try:
        if task == "research":
            summary = research_summarizer(raw_text)
            reply = "🔬 Research Summary:\n\n" + "\n".join(f"- {b}" for b in summary.split("\n"))

        elif task == "flashcards":
            flashcards = flashcard_agent.generate_flashcards(raw_text)  # Remove `st.session_state.flashcard_agent`
            st.session_state.generated_flashcards = flashcards
            reply = f"📚 Generated {len(flashcards)} flashcards from the document!"
            st.session_state.task = "flashcards"  # 🔥 Add this line


        elif task == "discharge":
            summary = discharge_summarizer(raw_text)
            reply = "🏥 Discharge Summary:\n\n" + format_discharge_summary(summary, extract_metadata(raw_text))

        elif task == "note":
            reply = "📝 Clinical Note:\n\n" + note_agent(raw_text)

        elif task == "quiz":
            mcqs = quiz_agent.generate(
                source_text=raw_text,
                mode=QuizMode.STUDENT,
                num_questions=5,
                difficulty="Understand"
            )
            st.session_state.generated_quiz = mcqs
            st.session_state.current_quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.task = "quiz"
            reply = f"🧠 Generated {len(mcqs)} quiz questions! Let's begin."

        elif task == "slides":
            with st.spinner("🛠 Generating slides..."):
                struct = slide_agent.create_structured_content(
                    raw_text, num_sections=5, style=SlideStyle.CLINICAL
                )
                pptx_file = slide_agent.create_presentation(struct, title="Chatbot Slides")
                
                # Save to session state for download
                st.session_state.generated_slides = pptx_file

                reply = "📊 Slides have been generated! Use the download button below to save your presentation."

                # Optional: show the structured content for transparency
                with st.expander("📋 Slide Structure Preview"):
                    st.code(struct, language="markdown")

                # Show the download button
                st.download_button(
                    label="⬇️ Download Slides (PPTX)",
                    data=pptx_file,
                    file_name="cellbot_slides.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )


        else:
            reply = "❌ Unknown task."

    except Exception as e:
        reply = f"❌ Error while processing file: {e}"

    # Append reply & clear pending task
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    save_chat_history(st.session_state.run_id, st.session_state.chat_history)
    st.session_state.pending_task = None
    st.session_state.pending_prompt = None

# 4) Chat history rendering
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], (bytes, bytearray)):
            st.download_button(
                "⬇️ Download", 
                msg["content"],
                key=f"file_{hash(msg['content'][:20])}"
            )
        else:
            # Improved markdown rendering with better formatting
            st.markdown(msg["content"], unsafe_allow_html=True)


if st.session_state.get("task") == "quiz":
    show_quiz()
    st.markdown("---")

if st.session_state.get("task") == "flashcards":
    show_flashcards()
    st.markdown("---")  # Optional separator

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ Guidance")

    task = st.session_state.get("task")

    if task == "slide_generator":
        st.markdown("""
        ### **Slide Creation Tips**
        - For clinical slides:
          - Focus on key practice points
          - Use [CR] markers for clinical relevance
          - Add evidence levels [L1-5]
        - For academic slides:
          - Include references
          - Highlight study designs
        - Optimal slide length: 5–7 bullet points
        """)
    elif task in ["clinical_note", "quiz_student", "quiz_practitioner", "quiz_patient"]:
        st.markdown("""
        ### **Clinical Workflow Tips**
        - For clinical notes:
          - Include SOAP elements (Subjective, Objective, Assessment, Plan)
          - Use standard medical terminology
          - Document medication doses and frequencies

        - For quizzes:
          - Use precise medical terminology
          - Include CDSCO codes for medications
          - Focus on Indian clinical guidelines
        """)
    elif task in ["discharge_summary", "research_summary"]:
        st.markdown("""
        ### **Document Requirements**
        - Hospital/patient identification
        - ICD-10 codes for diagnoses
        - Structured clinical documentation
        - Medication details with:
          - Drug names
          - Doses
          - Frequencies
          - Durations
        """)
    else:
        st.markdown("ℹ️ Select or upload a document to get started.")

    st.markdown("---")

    # ───────────── Load Previous Session ──────────────
    st.markdown("## 💾 Load Previous Session")

    available_sessions = []
    session_options = {}

    os.makedirs("session_data", exist_ok=True)
    for filename in os.listdir("session_data"):
        if filename.endswith(".json"):
            session_id = filename.replace(".json", "")
            try:
                with open(os.path.join("session_data", filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    topic = data.get("topic", "General") if isinstance(data, dict) else "General"
            except Exception:
                topic = "Corrupt"
            label = f"{topic.title()} ({session_id})"
            session_options[label] = session_id
            available_sessions.append(label)

    if session_options:
        selected_label = st.selectbox("Select Session", available_sessions, key="selected_session")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Load Session"):
                selected_session_id = session_options[selected_label]
                try:
                    with open(f"session_data/{selected_session_id}.json", "r", encoding="utf-8") as f:
                        session_data = json.load(f)

                    if isinstance(session_data, dict):
                        st.session_state.chat_history = session_data.get("chat_history", [])
                        st.session_state.uploaded_files = session_data.get("uploaded_files", [])
                        st.session_state.active_file_id = session_data.get("active_file_id")
                        st.session_state.run_id = selected_session_id
                        st.session_state.task = session_data.get("topic")

                        # Reinject files into vector store
                        for file in st.session_state.uploaded_files:
                            if "path" in file and os.path.exists(file["path"]):
                                with open(file["path"], "rb") as fbin:
                                    st.session_state.file_storage.store_file(
                                        fbin.read(),
                                        file["name"],
                                        metadata={"task": st.session_state.task},
                                        session_id=selected_session_id
                                    )

                        st.success("✅ Session loaded!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid session format.")
                except Exception as e:
                    st.error(f"❌ Failed to load session: {e}")

        with col2:
            if st.button("🗑 Delete Session"):
                selected_session_id = session_options[selected_label]
                try:
                    os.remove(f"session_data/{selected_session_id}.json")
                    st.success("🗑 Session deleted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not delete session: {e}")
    else:
        st.info("No saved sessions yet.")

    st.markdown("---")
    st.markdown("## 📂 Uploaded Files")

    uploaded_files = st.session_state.get("uploaded_files", [])
    active_file_id = st.session_state.get("active_file_id")

    if uploaded_files:
        if active_file_id:
            active_file = next((f for f in uploaded_files if f["id"] == active_file_id), None)
            if active_file:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"**Active**: {active_file['name']}")
                with col2:
                    if st.button("✖ Deselect", key="deselect_active", use_container_width=True):
                        st.session_state.active_file_id = None
                        st.rerun()

        for idx, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file['name']}", expanded=False):
                st.caption(f"Type: {file['type'].split('/')[-1].upper()}")
                st.caption(f"Size: {file['size']//1024} KB")
                st.caption(f"Uploaded: {file['uploaded_at']}")

                col1, col2 = st.columns([1, 1])
                with col1:
                    if active_file_id == file["id"]:
                        if st.button("✖ Deselect", key=f"deselect_{file['id']}_{idx}", use_container_width=True):
                            st.session_state.active_file_id = None
                            st.rerun()
                    else:
                        if st.button("🔍 Select", key=f"select_{file['id']}_{idx}", use_container_width=True):
                            st.session_state.active_file_id = file["id"]
                            st.rerun()

                with col2:
                    try:
                        with open(file["path"], "rb") as f:
                            file_data = f.read()
                        st.download_button(
                            "⬇️ Download",
                            file_data,
                            file_name=file["name"],
                            mime=file["type"],
                            use_container_width=True,
                            key=f"download_{file['id']}_{idx}"
                        )
                    except Exception as e:
                        st.error(f"❌ Could not load file: {str(e)}")
    else:
        st.info("No files uploaded yet.")
