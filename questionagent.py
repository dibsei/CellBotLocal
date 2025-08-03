#question_agent.py
import ollama
from typing import List
import re
import subprocess
import time

class QuestionAgent:
    def __init__(self):
        self._ensure_ollama_running()

    def _ensure_ollama_running(self):
        """Check if Ollama server is running, start it if not"""
        try:
            ollama.list()  # Simple API call to check connection
        except:
            try:
                # Try to start the Ollama server
                subprocess.Popen(['ollama', 'serve'], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                time.sleep(3)  # Give it time to start
            except FileNotFoundError:
                raise RuntimeError(
                    "Ollama not installed. Please install from https://ollama.ai/"
                    "\nAfter installation, run: `ollama pull neural-chat`"
                )

    def generate_questions(self, topic: str, num_questions: int = 15) -> List[str]:
        prompt = f"""You are a medical professor creating an exam. Generate exactly {num_questions} high-quality medical questions about {topic}.
        
        Strict Rules:
        1. Each question must be numbered (1., 2., etc.)
        2. Questions should be 1-2 sentences
        3. Focus on clinical applications
        4. Use Indian medical context
        5. All questions must end with a '?'
        6. No yes/no questions
        7. Return exactly {num_questions} questions

        Example Format:
        1. What are the first-line investigations for suspected tuberculosis?
        2. How does the mechanism of action of beta-blockers help in hypertension?

        Now generate {num_questions} questions about {topic}:
        """

        try:
            response = ollama.chat(
                model='neural-chat',  # Note: Corrected from your original 'neural-chat' spelling
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,
                    'num_predict': 800
                }
            )
            
            raw_text = response['message']['content']
            questions = self._clean_questions(raw_text, num_questions)
            
            # Ensure we got the requested number of questions
            if len(questions) < num_questions:
                raise RuntimeError(f"Only received {len(questions)} valid questions")
                
            return questions
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate questions: {str(e)}")

    def _clean_questions(self, text: str, expected_count: int) -> List[str]:
        questions = []
        current_question = ""

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Match numbered questions (1., 2., etc.)
            if re.match(r'^\d+\.', line):
                if current_question:
                    questions.append(current_question)
                current_question = re.sub(r'^\d+\.\s*', '', line)
            elif current_question:
                current_question += " " + line

        if current_question:
            questions.append(current_question)

        # Strict validation
        valid_questions = [
            q.strip() for q in questions[:expected_count]
            if q.endswith('?') 
            and len(q.split()) >= 4  # Minimum 4 words
            and not q.lower().startswith(('is ', 'are ', 'do ', 'does ', 'did '))  # No yes/no
        ]

        return valid_questions[:expected_count]