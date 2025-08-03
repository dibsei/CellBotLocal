#helpers.py
from typing import Dict
import re
import streamlit as st
import time
import json
from datetime import datetime
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import os


# Helper functions
def display_quiz(mcqs):
    """Display generated quiz questions with progress, feedback, and exports."""
    st.subheader("📝 Generated Questions")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, mcq in enumerate(mcqs, start=1):
        # Update progress
        pct = int(i / len(mcqs) * 100)
        progress_bar.progress(pct)
        status_text.text(f"Loading question {i}/{len(mcqs)}...")

        # Each question in an expander
        with st.expander(f"Q{i}: {mcq['question']}", expanded=False):
            # Show options as a radio
            key = f"q_{hash(mcq['question'])}"
            selected = st.radio("Options:", mcq["options"], key=key)

            if selected:
                correct = mcq["options"][mcq["answer"]]
                explanation = mcq.get("explanation", "")
                reference   = mcq.get("reference", "")

                if selected == correct:
                    st.success(f"✅ Correct!\n\n**Explanation:** {explanation}")
                else:
                    st.error(
                        f"❌ Incorrect.\n\n"
                        f"**Correct Answer:** {correct}\n\n"
                        f"**Explanation:** {explanation}"
                    )

                if reference:
                    st.info(f"📑 Source: {reference}")

        # tiny pause to ensure UI updates smoothly
        time.sleep(0.01)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Export as JSON",
            data=json.dumps(mcqs, indent=2),
            file_name="quiz.json",
            help="Structured format for applications"
        )
    with col2:
        # Build a CSV-style string
        lines = []
        for q in mcqs:
            opts = "|".join(q["options"])
            correct_opt = q["options"][q["answer"]]
            lines.append(f"{q['question']},{opts},{correct_opt}")
        st.download_button(
            "📊 Export as CSV",
            data="\n".join(lines),
            file_name="quiz.csv",
            help="Spreadsheet-friendly format"
        )


def perform_clinical_validation(summary):
    """Enhanced clinical validation with visual feedback"""
    content_checks = {
        "Diagnoses (ICD-10)": {
            "pattern": r'(?:diagnosis|dx)[\s:]+(.*?)(?=\n\s*\n|$)',
            "validator": lambda x: bool(re.findall(r'[A-Z]\d{2}(?:\.\d{1,2})?\b', x)),
            "help": "Must include ICD-10 codes (e.g., J18.9)"
        },
        "Treatment Details": {
            "pattern": r'(?:treatment|therapy)[\s:]+(.*?)(?=\n\s*\n|$)',
            "validator": lambda x: len(x.strip()) > 20,
            "help": "Should describe treatments given"
        },
        "Medications (CDSCO)": {
            "pattern": r'(?:medications?|drugs)[\s:]+(.*?)(?=\n\s*\n|$)',
            "validator": lambda x: bool(re.findall(r'\(CDSCO DR-\w+\)', x)),
            "help": "Requires CDSCO codes (e.g., DR-MET500)"
        }
    }

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Required Clinical Elements**")
        for name, check in content_checks.items():
            section_match = re.search(check["pattern"], summary, re.IGNORECASE | re.DOTALL)
            if section_match:
                content = section_match.group(1)
                if check["validator"](content):
                    st.markdown(f"""
                    <div class="success-box">
                        ✓ {name}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        ✗ {name} (Invalid)
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("How to fix"):
                        st.info(check["help"])
            else:
                st.markdown(f"""
                <div class="error-box">
                    ✗ {name} (Missing)
                </div>
                """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("**Medication Validation**")
        meds = validator.validate_medications(summary)
        
        if meds["invalid"]:
            st.markdown(f"""
            <div class="error-box">
                ❌ {len(meds['invalid'])} medication issues found
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Problematic entries"):
                for med in meds["invalid"][:3]:
                    st.warning(med)
            st.info("**Correct format:**\n`Tab Metformin 500mg PO BD (CDSCO DR-MET500) x5 days`")
        else:
            st.markdown("""
            <div class="success-box">
                ✓ All medications properly formatted
            </div>
            """, unsafe_allow_html=True)
            with st.expander("View medications"):
                for med in meds["valid"]:
                    st.markdown(f"- {med}")

# Keep your existing extract_metadata() and format_discharge_summary() functions here
def extract_metadata(text: str) -> dict:
    """Extracts hospital and patient details from input text and side‐form inputs"""
    metadata = {
        'hospital_name': None,
        'hospital_address': None,
        'contact_info': None,
        'gstin': None,
        'pan': None,
        'patient_name': None,
        'age': None,
        'sex': None,
        'patient_id': None,
        'uhid': None,
        'policy_number': None,
        'admit_date': None,
        'discharge_date': None,
        'physician_name': None
    }
    
    # Hospital name & address (two‐line header)
    m = re.search(
        r"HOSPITAL[\s:]*(.*?)\s*\n(.*?)\s*(?=\n\s*(?:PATIENT|UHID|$))",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        metadata['hospital_name']    = m.group(1).strip()
        metadata['hospital_address'] = m.group(2).strip()
    
    # Contact info (phone or email) – look for lines starting “Contact” or “Phone”
    m = re.search(r"^(?:Contact|Phone)[:\s]*(.+)$", text, re.MULTILINE|re.IGNORECASE)
    if m:
        metadata['contact_info'] = m.group(1).strip()
    
    # GSTIN & PAN
    m = re.search(r"GSTIN\s*[:=]?\s*([A-Z0-9]{15})", text, re.IGNORECASE)
    if m: metadata['gstin'] = m.group(1)
    m = re.search(r"PAN\s*[:=]?\s*([A-Z]{5}[0-9]{4}[A-Z])", text, re.IGNORECASE)
    if m: metadata['pan'] = m.group(1)
    
    # Patient name
    # Patient name
    m = re.search(r"(?:Patient\s*Name|Name\s*of\s*Patient|Name)[:\s]*(.+)", text, re.IGNORECASE)
    if m:
        metadata['patient_name'] = m.group(1).strip()

    # Fallback – line after "Patient Details"
    if not metadata['patient_name']:
        m = re.search(r"Patient\s*Details.*?\n\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            line = m.group(1).strip()
            if len(line.split()) >= 2:
                metadata['patient_name'] = line

    
    # Age/Sex combined or separate
    # Age/Sex combined or separate
    m = re.search(r"Age[/\s\-]+(\d+)\s*[/\s\-]+\s*(Male|Female|[MFUO])", text, re.IGNORECASE)
    if m:
        metadata['age'] = m.group(1).strip()
        metadata['sex'] = m.group(2).strip().capitalize()
    else:
        m = re.search(r"Age[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            metadata['age'] = m.group(1).strip()

        # Match "Sex" or "Gender" with M/F or Male/Female
        m = re.search(r"(?:Sex|Gender)[:\s]*(Male|Female|[MFUO])", text, re.IGNORECASE)
        if m:
            metadata['sex'] = m.group(1).strip().capitalize()

    
    # Patient ID / MRN / UHID
    m = re.search(r"(?:Patient ID|MRN|UHID)[:\s]*(\w+)", text, re.IGNORECASE)
    if m:
        metadata['patient_id'] = m.group(1).strip()
        metadata['uhid']       = m.group(1).strip()
    
    # Policy/Claim Number
    m = re.search(r"(?:Policy|Claim)\s*No\.?[:\s]*(\w+)", text, re.IGNORECASE)
    if m:
        metadata['policy_number'] = m.group(1).strip()
    
    # Dates: Admission and Discharge (DD/MM/YYYY)
    dates = re.findall(r"\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](20\d{2})\b", text)
    if len(dates) >= 1:
        metadata['admit_date'] = "/".join(dates[0])
    if len(dates) >= 2:
        metadata['discharge_date'] = "/".join(dates[1])
    
    # Physician name (line starting “Physician:” or “Doctor:”)
    m = re.search(r"^(?:Physician|Doctor)[:\s]*(.+)$", text, re.MULTILINE|re.IGNORECASE)
    if m:
        metadata['physician_name'] = m.group(1).strip()
    
    return metadata

def format_discharge_summary(raw_text: str, metadata: dict) -> str:
    """
    Formats a raw LLM discharge summary and metadata into the hospital’s
    standardized two‑page template in Markdown.
    """
    # Page header
    md = "# 🏥 Hospital Discharge Summary\n\n"
    md += f"**Hospital Name:** {metadata.get('hospital_name', '____________________')}\n\n"
    md += f"**Address:** {metadata.get('hospital_address', '____________________')}\n\n"
    md += f"**Contact Information:** {metadata.get('contact_info', '____________________')}\n\n"

    # Patient Details
    md += "## 🧑‍⚕️ Patient Details\n"
    md += f"- **Name:** {metadata.get('patient_name', '________________')}\n"
    md += f"- **Age/Sex:** {metadata.get('age', '____')}/{metadata.get('sex', '__')}\n"
    md += f"- **Patient ID / MRN:** {metadata.get('patient_id', metadata.get('uhid', '________'))}\n"
    md += f"- **Date of Admission:** {metadata.get('admit_date', 'DD/MM/YYYY')}\n"
    md += f"- **Date of Discharge:** {metadata.get('discharge_date', 'DD/MM/YYYY')}\n\n"

    # Admission Details
    md += "## 🏥 Admission Details\n"
    admission = re.search(r"reason for admission[:\-]?(.*?)(?:\n\n|$)", raw_text, re.IGNORECASE | re.DOTALL)
    mode = re.search(r"mode of admission[:\-]?(.*?)(?:\n\n|$)", raw_text, re.IGNORECASE | re.DOTALL)
    md += f"**Reason for Admission:**  \n{admission.group(1).strip() if admission else '____________________'}\n\n"
    md += f"**Mode of Admission:** {mode.group(1).strip() if mode else '____________________'}\n\n"

    # Clinical Summary
    md += "## 📋 Clinical Summary\n"
    # Take first paragraph of raw_text after “Clinical Summary” header if present
    cs = re.search(r"(?:clinical summary[:\-]?\s*)(.*?)(?=\n\n|\Z)", raw_text, re.IGNORECASE | re.DOTALL)
    md += f"{cs.group(1).strip() if cs else '[Enter concise summary here]'}\n\n"

    # Treatment Provided
    md += "## 💉 Treatment Provided\n"
    tp = re.search(r"(?:treatment provided[:\-]?\s*)(.*?)(?=\n\n|\Z)", raw_text, re.IGNORECASE | re.DOTALL)
    md += f"{tp.group(1).strip() if tp else '[List treatments, surgeries, medications, tests]'}\n\n"

    # Discharge Condition
    md += "## 🏁 Discharge Condition\n"
    dc = re.search(r"(?:discharge condition[:\-]?\s*)(.*?)(?=\n\n|\Z)", raw_text, re.IGNORECASE | re.DOTALL)
    md += f"{dc.group(1).strip() if dc else '[Describe condition at discharge]'}\n\n"

    # Medications
    md += "## 💊 Prescribed Medications at Discharge\n"
    meds = re.findall(r"\d+\.\s*(.*?)(?=\n|$)", raw_text)
    if meds:
        for i, m in enumerate(meds, 1):
            md += f"{i}. {m.strip()}\n"
    else:
        md += "1. [Medication name, dose, frequency, duration]\n2. […]\n\n"

    # Follow-Up Instructions
    md += "\n## 🔄 Follow‑Up Instructions\n"
    fu = re.search(r"(?:follow[- ]up instructions[:\-]?\s*)(.*?)(?=\n\n|\Z)", raw_text, re.IGNORECASE | re.DOTALL)
    if fu:
        for line in fu.group(1).split("\n"):
            line = line.strip()
            if line:
                md += f"- {line}\n"
    else:
        md += "- Recommended Follow‑Up: ____________________\n"
        md += "- Next Follow‑Up Date: ____________________\n"
        md += "- Lifestyle/Dietary Instructions: ____________________\n"

    # Physician & Patient Signatures
    md += "\n---\n"
    md += "**Physician's Name & Signature:**  \n"
    md += f"{metadata.get('physician_name', '____________________')}  \n\n"
    md += "**Patient/Guardian Acknowledgment:**  \n"
    md += "I acknowledge that I have received and understood the above information.  \n"
    md += "Patient / Guardian Signature: ____________________  \n"
    md += f"Date: {datetime.now():%d/%m/%Y}\n\n"

    # Disclaimer
    md += "_Disclaimer: This is a standardized template. Modify per hospital/NABH guidelines._\n"

    return md


def detect_task(prompt: str) -> str:
    """Detect task type from user prompt"""
    prompt = prompt.lower()
    if any(x in prompt for x in ["summarize", "research paper", "study summary"]):
        return "research"
    elif "discharge" in prompt or "insurance" in prompt:
        return "discharge"
    elif "clinical note" in prompt or "soap" in prompt:
        return "note"
    elif "quiz" in prompt or "mcq" in prompt:
        return "quiz"
    elif "slide" in prompt or "presentation" in prompt:
        return "slides"
    elif "flashcard" in prompt or "flashcards" in prompt:
        return "flashcards"
    else:
        return "unknown"

def neural_chat_response(prompt: str) -> str:
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "neural-chat", "prompt": prompt, "stream": False}
        )
        return res.json()["response"]
    except Exception as e:
        return f"❌ Neural-Chat error: {e}"
    
def extract_text_from_image(image_path_or_pil) -> str:
    try:
        # Load model and processor
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        # Open image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        # Prepare inputs
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generate output
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

    except Exception as e:
        return f"❌ TrOCR error: {str(e)}"
    
def show_flashcards():
    if not st.session_state.generated_flashcards:
        st.warning("No flashcards generated yet. Upload a document and ask for flashcards.")
        return
    
    st.markdown("### 📚 Medical Flashcards")
    st.caption(f"Card {st.session_state.current_flashcard + 1} of {len(st.session_state.generated_flashcards)}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅ Previous", disabled=st.session_state.current_flashcard == 0):
            st.session_state.current_flashcard -= 1
            st.session_state.show_flashcard_answer = False
    with col3:
        if st.button("➡ Next", disabled=st.session_state.current_flashcard >= len(st.session_state.generated_flashcards) - 1):
            st.session_state.current_flashcard += 1
            st.session_state.show_flashcard_answer = False
    
    # Current flashcard
    card = st.session_state.generated_flashcards[st.session_state.current_flashcard]
    
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0;">
        <h3>❓ {card['front']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.show_flashcard_answer:
        st.markdown(f"""
        <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f0f8ff;">
            <h3>💡 {card['back']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔍 Reveal Answer", disabled=st.session_state.show_flashcard_answer):
        st.session_state.show_flashcard_answer = True
    
    # Progress
    progress = (st.session_state.current_flashcard + 1) / len(st.session_state.generated_flashcards)
    st.progress(progress)

def save_chat_history(run_id: str, history: list):
    os.makedirs("session_data", exist_ok=True)
    with open(f"session_data/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history(run_id: str) -> list:
    try:
        with open(f"session_data/{run_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def show_quiz():
    if not st.session_state.generated_quiz:
        st.info("No quiz questions generated yet.")
        return

    i = st.session_state.current_quiz_index
    if i >= len(st.session_state.generated_quiz):
        st.success(f"🏁 Quiz Complete! You scored {st.session_state.quiz_score}/{len(st.session_state.generated_quiz)}")
        return

    q = st.session_state.generated_quiz[i]
    st.markdown(f"**Q{i+1}. {q['question']}**")

    selected = st.radio("Choose an option:", q["options"], key=f"quiz_q{i}", index=None)

    if st.button("Submit", key=f"submit_q{i}"):
        if i in st.session_state.answered_quiz:
            st.warning("You've already answered this question.")
        else:
            st.session_state.answered_quiz.add(i)
            correct_idx = q["answer"]
            correct_option = q["options"][correct_idx]
            if selected == correct_option:
                st.success("✅ Correct!")
                st.session_state.quiz_score += 1
            else:
                st.error(f"❌ Incorrect. Correct answer: **{correct_option}**")
            st.info(f"🧠 Explanation: {q.get('explanation', 'N/A')}")
            if q.get("reference"):
                st.caption(f"📄 Reference: _{q['reference']}_")

    if i in st.session_state.answered_quiz:
        if st.button("Next", key=f"next_q{i}"):
            st.session_state.current_quiz_index += 1
            st.rerun()
