#summarizer.py
from langchain_community.llms import Ollama
from typing import List, Callable
import re
from enum import Enum
import json 

class SummaryType(Enum):
    RESEARCH = "research"
    DISCHARGE = "discharge"

def build_medical_summarizer(summary_type: SummaryType = SummaryType.RESEARCH):
    """Builds either research or discharge summarizer"""
    llm = Ollama(
        model="neural-chat",
        temperature=0.3 if summary_type == SummaryType.RESEARCH else 0.1,  # Lower temp for clinical
        system=_get_system_prompt(summary_type)
    )

    if summary_type == SummaryType.RESEARCH:
        return _create_research_summarizer(llm)
    else:
        return _create_discharge_summarizer(llm)

def _get_system_prompt(summary_type: SummaryType) -> str:
    """Returns appropriate system prompt for research or discharge."""
    if summary_type == SummaryType.RESEARCH:
        return """
You are a senior biomedical researcher. Read the full medical research paper and summarize it in exactly 6 bullet points with labels and short descriptions.

Output Format:
1. [MAIN TOPIC] - (Main topic of the study)
2. [KEY FINDING] - (Most important result)
3. [METHODS] - (Study design or approach)
4. [POPULATION] - (Sample or subjects used)
5. [FINAL FINDING] - (Clinical or biological conclusion)
6. [CONCLUSION] - (Outcomes or results)

Rules:
- Use precise medical terminology
- Each bullet must be > 20 words
- Do NOT skip any bullet
- If information is missing, make a best guess from context
- **STRICTLY** follow the output format with all 6 bullets.
- If a conclusion is missing, write: `5. [FINAL FINDING] - [Not explicitly stated in the paper]
"""
    else:
        # Discharge summary prompt now requests JSON for downstream parsing
        return """
You are a hospital discharge summary generator.  From the clinical notes provided,
produce a JSON object with the following keys (omit any key if no relevant data exists):

- reason_for_admission
- mode_of_admission
- clinical_summary
- treatment_provided
- discharge_condition
- prescribed_medications
- follow_up_instructions
- physician_name
- patient_acknowledgment

Output ONLY valid JSON.  Example:
{
  "reason_for_admission": "Patient presented with ...",
  "mode_of_admission": "Emergency via ambulance",
  "clinical_summary": "On exam ...",
  "treatment_provided": "1. Drug A ...\\n2. ...",
  "discharge_condition": "Stable, ambulating ...",
  "prescribed_medications": "1. Aspirin 75mg ...\\n2. ...",
  "follow_up_instructions": "- low salt diet\\n- follow‑up in 1 month",
  "physician_name": "Dr. X",
  "patient_acknowledgment": "I acknowledge ..."
}

Rules:
- Omit keys if no information is present in the notes.
- Do NOT include any additional keys.
- Ensure valid JSON syntax.
"""


def _create_research_summarizer(llm) -> Callable[[str], str]:
    """Your existing research bullet-point summarizer"""
    def enforce_bullets(text: str) -> List[str]:
        required_labels = ["MAIN TOPIC", "KEY FINDING", "METHODS", "POPULATION", "CONCLUSION"]
        bullets = []
    
    # Extract valid bullets
        for line in text.strip().split('\n'):
            if re.match(r'^\d\.\s*\[.+?\]\s*-\s*.+', line.strip()):
                bullets.append(line.strip())
    
    # Check if all required labels are present
        found_labels = [re.search(r'\[(.*?)\]', bullet).group(1) for bullet in bullets if re.search(r'\[(.*?)\]', bullet)]
        missing_labels = [label for label in required_labels if label not in found_labels]
    
    # Add missing ones
        for label in missing_labels:
            bullets.append(f"{len(bullets)+1}. [{label}] - [Content not available]")
    
        return bullets[:6]  # Ensure exactly 5 bullets


    def summarize(text: str) -> str:
        response = llm.invoke(f"Summarize this medical paper:\n{text[:10000]}")
        return "\n".join(enforce_bullets(response))

    return summarize

    

def _create_discharge_summarizer(llm) -> Callable[[str], str]:
    """
    New discharge summary generator that returns a text
    with explicit section headers for the formatter to parse.
    """
    def summarize(clinical_notes: str) -> str:
        prompt = f"""
You are a hospital discharge summary generator.  From the clinical notes below,
produce a JSON object with the following keys (omit keys if no information exists):

1. reason_for_admission
2. mode_of_admission
3. clinical_summary
4. treatment_provided
5. discharge_condition
6. prescribed_medications
7. follow_up_instructions
8. physician_name
9. patient_acknowledgment

Output ONLY valid JSON.  Example:
{{
  "reason_for_admission": "Patient presented with ...",
  "mode_of_admission": "Emergency via ambulance",
  "clinical_summary": "On exam ...",
  "treatment_provided": "1. Drug A ...\\n2. ...",
  "discharge_condition": "Stable, ambulating ...",
  "prescribed_medications": "1. Aspirin 75mg ...\\n2. ...",
  "follow_up_instructions": "- low salt diet\\n- follow‑up in 1 month",
  "physician_name": "Dr. X",
  "patient_acknowledgment": "I acknowledge ..."
}}

Clinical Notes:
\"\"\"
{clinical_notes[:15000]}
\"\"\"
"""
        # Invoke the LLM
        resp = llm.invoke(prompt)
        text = resp if isinstance(resp, str) else resp.get("message", resp)

        # Extract JSON blob
                # Extract JSON blob
        m = re.search(r'```json\n([\s\S]*?)\n```', text)
        blob = m.group(1) if m else text.strip()

        def _clean_json(text: str) -> str:
            return re.sub(r',\s*([\]}])', r'\1', text)

        try:
            clean_blob = _clean_json(blob)
            data = json.loads(clean_blob)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse discharge JSON: {e}\nOriginal:\n{text}")

        # Reconstruct into a text with section headers your formatter can read
        out = []
        if data.get("reason_for_admission"):
            out.append(f"Reason for admission: {data['reason_for_admission']}")
        if data.get("mode_of_admission"):
            out.append(f"Mode of admission: {data['mode_of_admission']}")
        if data.get("clinical_summary"):
            out.append(f"Clinical summary: {data['clinical_summary']}")
        if data.get("treatment_provided"):
            out.append(f"Treatment provided: {data['treatment_provided']}")
        if data.get("discharge_condition"):
            out.append(f"Discharge condition: {data['discharge_condition']}")
        if data.get("prescribed_medications"):
            out.append(f"Prescribed medications at discharge: {data['prescribed_medications']}")
        if data.get("follow_up_instructions"):
            out.append(f"Follow-up instructions: {data['follow_up_instructions']}")
        if data.get("physician_name"):
            out.append(f"Physician: {data['physician_name']}")
        if data.get("patient_acknowledgment"):
            out.append(f"Patient Acknowledgment: {data['patient_acknowledgment']}")

        return "\n\n".join(out)

    return summarize
