#document_processor.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from typing import List, Union
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self):
        # Reusable text splitter for both research and clinical notes
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    
    def process_pdf(self, uploaded_file) -> List[Document]:
        """Process PDFs for both research and clinical notes.
        Args:
            uploaded_file: Streamlit UploadedFile object
        Returns:
            List of LangChain Documents with metadata (page numbers, source)
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load_and_split(self.text_splitter)
            
            # Add document type metadata (useful for routing later)
            for page in pages:
                page.metadata["doc_type"] = self._infer_document_type(page.page_content)
            return pages
            
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _infer_document_type(self, text: str) -> str:
        """Heuristic to detect if document is clinical notes vs. research paper.
        Returns:
            "clinical" or "research"
        """
        text_lower = text.lower()
        clinical_keywords = {"patient", "diagnosis", "discharge", "medication", "follow-up"}
        research_keywords = {"abstract", "introduction", "methodology", "results"}
        
        if any(keyword in text_lower for keyword in clinical_keywords):
            return "clinical"
        elif any(keyword in text_lower for keyword in research_keywords):
            return "research"
        return "unknown"  # Fallback