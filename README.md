##  CellBot Medical Assistant

A modular Streamlit-based LLM app for:
- 📄 Summarizing medical documents
- 📝 Generating clinical notes
- ❓ Creating flashcards and quizzes
- 🧾 Writing discharge summaries
- 📊 Making slide presentations

##  Project Structure

```plaintext
CellBot/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── .env                      # Environment variables

├── agents/                   # All LLM agent logic
│   ├── __init__.py
│   ├── flashcards.py         # FlashcardAgent
│   ├── note_agent.py         # Clinical notes generator
│   ├── questionagent.py      # (Optional) Custom Q&A agent
│   ├── quiz_agent.py         # Quiz MCQ generator
│   ├── slide_agent.py        # Slide generator (PPTX)
│   ├── summarizer.py         # Research/discharge summarizer

├── utils/                    # Helper utilities
│   ├── __init__.py
│   ├── clinical_validator.py # Rule-based clinical checks
│   ├── document_processor.py # PDF/Image text extraction
│   ├── metadata_utils.py     # Metadata extraction

├── uploads/                  # Uploaded PDFs, images
├── session_data/             # Saved chat/file sessions (JSON)

├── helpers.py                # UI + OCR helper functions
├── rag_engine.py             # RAG-based QA engine
├── task_router.py            # (Optional) Task intent routing
└── venv/                     # Python virtual environment

