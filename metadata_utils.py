#metadata_utils.py
import re
from typing import Dict
from datetime import datetime

def extract_metadata(text: str) -> Dict:
    """Enhanced metadata extraction using patterns from ClinicalValidator"""
    metadata = {
        'hospital_name': None,
        'hospital_address': None,
        'gstin': None,
        'pan': None,
        'patient_name': None,
        'uhid': None,
        'policy_number': None,
        'admit_date': None,
        'discharge_date': None,
        'topic': None
    }
    
    # Extract hospital info
    hospital_match = re.search(r"HOSPITAL[\s:]*(.*?)(?:\n|$)(.*?)(?=\n\s*(?:PATIENT|UHID|$))", 
                             text, re.IGNORECASE | re.DOTALL)
    if hospital_match:
        metadata.update({
            'hospital_name': hospital_match.group(1).strip(),
            'hospital_address': hospital_match.group(2).strip()
        })
    
    # Extract using ClinicalValidator patterns (reuse the regex patterns)
    metadata.update({
        'gstin': re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1}\b', text),
        'pan': re.search(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b', text),
        'uhid': re.search(r'\bUHID\s*:\s*[A-Z0-9]{8,12}\b', text),
        'policy_number': re.search(r'\b(?:Policy|Insurance)\s*No\.?\s*:\s*[A-Z0-9]{10,15}\b', text)
    })
    
    # Extract dates (Indian format)
    date_matches = re.findall(r'\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[012])[-/](20\d{2})\b', text)
    if len(date_matches) >= 2:
        metadata.update({
            'admit_date': "/".join(date_matches[0]),
            'discharge_date': "/".join(date_matches[1])
        })
    
    return metadata