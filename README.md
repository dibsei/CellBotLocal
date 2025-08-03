## 🧠 CellBot Medical Assistant

A modular Streamlit-based LLM app for:
- 📄 Summarizing medical documents
- 📝 Generating clinical notes
- ❓ Creating flashcards and quizzes
- 🧾 Writing discharge summaries
- 📊 Making slide presentations

### 📁 Project Structure
CellBot/
├── app.py                       # Main Streamlit app
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation (add this)
├── .env                        # Environment variables (if any)

├── agents/                     # All agent-based logic (LLM chains)
│   ├── __init__.py
│   ├── flashcards.py           # FlashcardAgent for spaced repetition
│   ├── note_agent.py           # Generates clinical notes
│   ├── questionagent.py        # [Optional] Unused or under development
│   ├── quiz_agent.py           # MCQAgent for generating quizzes
│   ├── slide_agent.py          # SlideAgent for creating PPTX slides
│   ├── summarizer.py           # Builds research/discharge summarizers

├── utils/                      # Utility functions and helpers
│   ├── __init__.py
│   ├── clinical_validator.py   # Clinical validation and rule checks
│   ├── document_processor.py   # Extracts text from PDFs, images, etc.
│   ├── metadata_utils.py       # Extracts structured metadata from content

├── uploads/                    # Uploaded files (PDFs, images)
├── session_data/               # Saved chat + file sessions (JSON)

├── helpers.py                  # Shared UI functions, OCR, and rendering
├── rag_engine.py               # Retrieval-Augmented Generation engine
├── task_router.py              # [Optional] Route task based on input intent
└── venv/                       # Python virtual environment (ignored in git)
