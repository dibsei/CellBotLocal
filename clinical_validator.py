#clinical_validator.py 
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class ClinicalValidator:
    """Validates clinical content for discharge summaries including Indian insurance requirements"""
    
    # Enhanced medication pattern with Indian drug codes and formulations
    MEDICATION_PATTERN = r"""
        ^\s*[\-\*]?\s*                          # Optional bullet
        (?P<name>[A-Z][A-Za-z\-\s/]+?)          # Medication name
        \s+
        (?P<dose>\d+\.?\d*)\s*                  # Dose
        (?P<unit>mg|mcg|g|mL|units?)\s+         # Unit
        (?P<route>PO|IV|IM|SC|PR|INH|TOP|SL)\s+ # Route (added TOP and SL)
        (?P<frequency>q?\d*[h]?|daily|BD|TDS|QID|SOS|PRN)  # Frequency (Indian terminology)
        (?:\s*,\s*                              # Optional details
            (?:(?P<duration>\d+\s*(?:day|week|month)s?)\s*)?
            (?:(?:for\s*(?P<indication>[^\n,]+))?\s*)?
            (?:\((?P<drug_code>CDSCO\s*[A-Z0-9-]+)\)\s*)?  # Indian drug code
        )?
        \s*$
    """
    
    # Indian insurance-required patterns
    ICD10_PATTERN = r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
    DISCHARGE_STATUS_PATTERN = r'\bDischarge\s*(?:status|disposition).*?(?:Improved|Recovered|Referred|Absconded|LAMA|DAMA|Expired)\b'
    UHID_PATTERN = r'\bUHID\s*:\s*[A-Z0-9]{8,12}\b'
    POLICY_NUMBER_PATTERN = r'\b(?:Policy|Insurance)\s*No\.?\s*:\s*[A-Z0-9]{10,15}\b'
    PREAUTH_NUMBER_PATTERN = r'\bPre[- ]?auth\s*(?:No\.?|Number)\s*:\s*[A-Z0-9]{8,12}\b'
    PAN_PATTERN = r'\b[A-Z]{5}\d{4}[A-Z]{1}\b'
    GSTIN_PATTERN = r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1}\b'
    INDIAN_DATE_PATTERN = r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[012])[-/](?:20\d{2})\b'

    def validate_medications(self, text: str) -> Dict[str, List]:
        """
        Validates medication formats with Indian insurance requirements.
        Returns {'valid': [], 'invalid': []}
        """
        results = {"valid": [], "invalid": []}
        
        # Find medication section using flexible matching
        med_section_match = re.search(
            r'(?i)(?:medications?|meds|discharge\s+prescriptions?|treatment\s+given)[:\s]*(.*?)(?=\n\s*\n|\Z)',
            text, re.DOTALL)
        
        if not med_section_match:
            return results
            
        med_section = med_section_match.group(1)
        
        # Check each medication line
        for line in med_section.split('\n'):
            line = line.strip()
            if not line or line.startswith(('#', '//')):
                continue
                
            # Standardize Indian medical abbreviations
            line = (line.replace("OD", "daily")
                    .replace("BD", "BID")
                    .replace("TDS", "TID")
                    .replace("QDS", "QID")
                    .replace("SOS", "PRN")
                    .replace("nocte", "at bedtime"))
            
            match = re.fullmatch(self.MEDICATION_PATTERN, line, re.VERBOSE | re.IGNORECASE)
            if match:
                med_str = f"{match.group('name')} {match.group('dose')}{match.group('unit')} {match.group('route')} {match.group('frequency')}"
                if match.group('duration'):
                    med_str += f" x{match.group('duration')}"
                if match.group('drug_code'):
                    med_str += f" (CDSCO {match.group('drug_code')})"
                results["valid"].append(med_str)
            else:
                results["invalid"].append(line)
                
        return results

    def validate_insurance_requirements(self, text: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Checks for Indian insurance-mandated documentation elements.
        Returns {requirement: (is_present, extra_info)}
        """
        results = {}
        
        # ICD-10 Codes (required by all Indian insurers)
        icd_codes = re.findall(self.ICD10_PATTERN, text)
        results["ICD-10 Codes"] = (bool(icd_codes), f"Found: {', '.join(icd_codes)}" if icd_codes else None)
        
        # Unique Health ID (UHID)
        uhid_match = re.search(self.UHID_PATTERN, text)
        results["UHID"] = (bool(uhid_match), uhid_match.group(0) if uhid_match else None)
        
        # Policy Number
        policy_match = re.search(self.POLICY_NUMBER_PATTERN, text, re.IGNORECASE)
        results["Policy Number"] = (bool(policy_match), policy_match.group(0) if policy_match else None)
        
        # Pre-authorization Number
        preauth_match = re.search(self.PREAUTH_NUMBER_PATTERN, text, re.IGNORECASE)
        results["Pre-auth Number"] = (bool(preauth_match), preauth_match.group(0) if preauth_match else None)
        
        # Discharge Status (Indian specific terms)
        status_match = re.search(self.DISCHARGE_STATUS_PATTERN, text, re.IGNORECASE)
        results["Discharge Status"] = (
            bool(status_match), 
            status_match.group(0) if status_match else None
        )
        
        # Dates validation (admission and discharge)
        date_matches = re.findall(self.INDIAN_DATE_PATTERN, text)
        results["Admit/Discharge Dates"] = (
            len(date_matches) >= 2,
            f"Found: {', '.join(date_matches[:2])}" if date_matches else None
        )
        
        # Hospital PAN and GSTIN (for cashless claims)
        pan_match = re.search(self.PAN_PATTERN, text)
        results["Hospital PAN"] = (bool(pan_match), pan_match.group(0) if pan_match else None)
        
        gstin_match = re.search(self.GSTIN_PATTERN, text)
        results["Hospital GSTIN"] = (bool(gstin_match), gstin_match.group(0) if gstin_match else None)
        
        # Validate date sequence (admission before discharge)
        if len(date_matches) >= 2:
            try:
                admit_date = datetime.strptime(date_matches[0], "%d/%m/%Y")
                discharge_date = datetime.strptime(date_matches[1], "%d/%m/%Y")
                results["Date Sequence Valid"] = (admit_date <= discharge_date, 
                                                f"Admit: {date_matches[0]}, Discharge: {date_matches[1]}")
            except ValueError:
                results["Date Sequence Valid"] = (False, "Invalid date format")
        
        return results

    def get_insurance_validation_report(self, text: str) -> str:
        """Generates a human-readable insurance compliance report for Indian insurers"""
        report = ["DISCHARGE SUMMARY COMPLIANCE REPORT (INDIAN INSURANCE STANDARDS)", "="*60]
        checks = self.validate_insurance_requirements(text)
        
        # Critical requirements
        report.append("\nCRITICAL REQUIREMENTS:")
        critical_items = ["ICD-10 Codes", "UHID", "Policy Number", "Discharge Status", "Admit/Discharge Dates"]
        for req in critical_items:
            is_present, detail = checks.get(req, (False, None))
            status = "✓" if is_present else "✗"
            report.append(f"{status} {req}")
            if detail:
                report.append(f"   → {detail}")
        
        # Additional requirements
        report.append("\nADDITIONAL REQUIREMENTS:")
        additional_items = ["Pre-auth Number", "Hospital PAN", "Hospital GSTIN", "Date Sequence Valid"]
        for req in additional_items:
            is_present, detail = checks.get(req, (False, None))
            status = "✓" if is_present else "✗"
            report.append(f"{status} {req}")
            if detail:
                report.append(f"   → {detail}")
        
        # Medications validation
        med_results = self.validate_medications(text)
        report.append("\nMEDICATION VALIDATION:")
        report.append(f"✓ Valid medications: {len(med_results['valid'])}")
        report.append(f"✗ Invalid medication formats: {len(med_results['invalid'])}")
        if med_results['invalid']:
            report.append("   Invalid medication entries:")
            for med in med_results['invalid'][:3]:  # Show first 3 invalid for brevity
                report.append(f"   → {med}")
            if len(med_results['invalid']) > 3:
                report.append(f"   (...and {len(med_results['invalid'])-3} more)")
        
        return "\n".join(report)

    def is_insurance_compliant(self, text: str) -> bool:
        """Determines if the discharge summary meets minimum insurance requirements"""
        checks = self.validate_insurance_requirements(text)
        critical_items = ["ICD-10 Codes", "UHID", "Policy Number", "Discharge Status", "Admit/Discharge Dates"]
        return all(checks.get(item, (False, None))[0] for item in critical_items)