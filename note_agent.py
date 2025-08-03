#note_agent.py
from langchain_community.llms import Ollama
from typing import List, Callable
import re
from enum import Enum

class NoteType(Enum):
    CLINICAL = "clinical"
    RESEARCH = "research"

def build_note_agent(note_type: NoteType = NoteType.CLINICAL):
    """Builds either clinical or research note agent"""
    llm = Ollama(
        model="neural-chat",
        temperature=0.2,  # Lower temp for accuracy
        system=_get_note_system_prompt(note_type)
    )

    if note_type == NoteType.CLINICAL:
        return _create_clinical_note_agent(llm)
    else:
        return _create_research_note_agent(llm)

def _get_note_system_prompt(note_type: NoteType) -> str:
    if note_type == NoteType.CLINICAL:
        return """
        You are a medical scribe. Convert input into structured clinical notes:
        - **Subjective**: Patient-reported details
        - **Objective**: Vitals, labs, findings
        - **Assessment**: Diagnosis/differential
        - **Plan**: Next steps
        
        Rules:
        1. Never invent information
        2. Use medical abbreviations (e.g., "SOB" for shortness of breath)
        3. Keep each section under 50 words
        """
    else:
        return """
        You are a research assistant. Convert input into structured notes:
        1. [TOPIC] - Main subject
        2. [KEY POINTS] - 3-5 concepts
        3. [CITATIONS] - Relevant papers/studies
        4. [QUESTIONS] - Unanswered research questions
        
        Rules:
        1. Use bullet points
        2. Highlight contradictory evidence if present
        """

def _create_clinical_note_agent(llm) -> Callable[[str], str]:
    def enforce_soap_format(text: str) -> str:
        """Strict SOAP enforcement for Indian clinical standards"""
        sections = {
            "Subjective": r"(?i)(?:history|subjective|complaints?)[:\s]*(.*?)(?=\n\s*(?:objective|exam|$))",
            "Objective": r"(?i)(?:objective|exam|findings)[:\s]*(.*?)(?=\n\s*(?:assessment|impression|$))",
            "Assessment": r"(?i)(?:assessment|impression|diagnosis)[:\s]*(.*?)(?=\n\s*(?:plan|advice|$))",
            "Plan": r"(?i)(?:plan|advice|rx)[:\s]*(.*)"
        }
        
        note = ""
        for section, pattern in sections.items():
            match = re.search(pattern, text, re.DOTALL)
            content = match.group(1).strip() if match else "[Not documented]"
            
            # Special processing for Indian context
            if section == "Plan" and "rx" in text.lower():
                content = re.sub(r"(?i)rx\s*[:]?\s*", "", content)
                content = re.sub(r"tab\s*", "Tablet ", content)
            
            note += f"**{section}**:\n{content}\n\n"
        return note.strip()

    def create_note(text: str) -> str:
        """Pre-process Indian clinical text before LLM"""
        # Standardize common Indian abbreviations
        replacements = {
            r"\bsob\b": "shortness of breath",
            r"\blt\b": "left",
            r"\brt\b": "right",
            r"\bdm\b": "diabetes mellitus",
            r"\bhtn\b": "hypertension",
            r"\btab\b": "tablet"
        }
        
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
        response = llm.invoke(f"Create SOAP note from Indian clinical text:\n{text[:8000]}")
        return enforce_soap_format(response)
    
    return create_note

def _create_research_note_agent(llm) -> Callable[[str], str]:
    def enforce_research_structure(text: str) -> str:
        """Ensures 4-section research note format"""
        required = ["[TOPIC]", "[KEY POINTS]", "[CITATIONS]", "[QUESTIONS]"]
        for marker in required:
            if marker not in text:
                text += f"\n{marker} - [Not specified]"
        return text

    def create_note(text: str) -> str:
        response = llm.invoke(f"Create research note from:\n{text[:10000]}")
        return enforce_research_structure(response)
    
    return create_note