# agents/flashcards.py
from typing import List, Dict
from langchain_community.llms import Ollama
import json

class FlashcardAgent:
    def __init__(self):
        self.llm = Ollama(
            model="neural-chat",
            temperature=0.2,  # Lower temp for factual accuracy
            system="""
            You are a medical flashcard generator. Create concise flashcards from medical content.
            Each flashcard should have:
            - Front: A clear question or term
            - Back: Precise medical definition or explanation
            
            Rules:
            1. Use standard medical terminology
            2. Include important details like doses when relevant
            3. Keep answers under 50 words
            4. Format as valid JSON
            """
        )

    def generate_flashcards(self, content: str, num_cards: int = 10) -> List[Dict[str, str]]:
        """Generate flashcards from medical content"""
        prompt = f"""
        From this medical content, create {num_cards} high-yield flashcards:
        
        {content[:10000]}  # Truncate to prevent overload
        
        Output format (JSON):
        {{
            "flashcards": [
                {{
                    "front": "question or term",
                    "back": "answer or definition"
                }},
                ...
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response)
            return data.get("flashcards", [])
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return []