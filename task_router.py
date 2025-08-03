# task_router.py

import re

def detect_task(prompt: str) -> str:
    prompt = prompt.lower()

    if any(x in prompt for x in ["quiz", "mcq", "multiple choice"]):
        if "patient" in prompt:
            return "quiz_patient"
        elif "student" in prompt:
            return "quiz_student"
        elif "practitioner" in prompt or "doctor" in prompt:
            return "quiz_practitioner"
        else:
            return "quiz_student"  # default

    elif any(x in prompt for x in ["discharge summary", "hospital discharge"]):
        return "discharge_summary"

    elif any(x in prompt for x in ["research summary", "summarize paper", "journal", "article"]):
        return "research_summary"

    elif any(x in prompt for x in ["clinical note", "case note", "medical note"]):
        return "clinical_note"

    elif any(x in prompt for x in ["slides", "presentation", "ppt"]):
        return "slide_generator"

    else:
        return "unknown"
