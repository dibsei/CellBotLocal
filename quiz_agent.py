# agents/quiz_agent.py

from enum import Enum
from typing import List, Dict
import json, re
from langchain_community.llms import Ollama


class QuizMode(Enum):
    STUDENT = ("student", "Medical Student Education")
    PRACTITIONER = ("practitioner", "Memory Reinforcement for Practitioners")

    def __init__(self, value, label):
        self._value_ = value
        self.label = label


class MCQAgent:
    def __init__(self):
        self.llm = Ollama(model="neural-chat", temperature=0.2)

    def generate(
        self,
        source_text: str,
        mode: QuizMode,
        num_questions: int = 10,
        difficulty: str = "Recall",
        chat_name: str = "General"
    ) -> List[Dict]:
        system_prompt = self._system_prompt(mode)
        instruction = self._difficulty_instruction(difficulty)

        user_prompt = f"""
You are now assisting in the quiz session titled '{chat_name}'.

{system_prompt}

Context:
\"\"\"
{source_text.strip()}
\"\"\"

Instruction:
{instruction}

Generate exactly {num_questions} MCQs in JSON array format:
[
  {{
    "question": "...",
    "options": ["...","...","...","..."],
    "answer": 0,
    "explanation": "..."
  }}
]
"""
        response = self.llm.invoke(user_prompt)
        text = response if isinstance(response, str) else response.get("message", "")
        mcqs = self._parse(text)
        return self._validate(mcqs, source_text, num_questions)

    def _system_prompt(self, mode: QuizMode) -> str:
        if mode == QuizMode.STUDENT:
            return "You are a medical professor generating MCQs for medical students."
        elif mode == QuizMode.PRACTITIONER:
            return "You are a clinical expert generating memory-reinforcement MCQs."
        return ""

    def _difficulty_instruction(self, level: str) -> str:
        return {
            "Recall": "Ask fact-based recall questions.",
            "Understand": "Ask conceptual understanding questions.",
            "Apply": "Ask scenario-based application questions."
        }.get(level, "")

    def _parse(self, text: str) -> List[Dict]:
        try:
            blob = re.search(r'```json\n([\s\S]*?)\n```', text)
            return json.loads(blob.group(1)) if blob else json.loads(text)
        except:
            mcqs = []
            for match in re.finditer(r'\{[^}]*"question"[^}]*\}', text):
                try:
                    mcqs.append(json.loads(match.group()))
                except:
                    continue
            return mcqs

    def _validate(self, mcqs: List[Dict], context: str, limit: int) -> List[Dict]:
        valid = []
        for q in mcqs:
            if (
                isinstance(q.get("question"), str)
                and isinstance(q.get("options"), list) and len(q["options"]) == 4
                and isinstance(q.get("answer"), int)
                and q.get("explanation")
            ):
                for sentence in context.split("."):
                    if q["explanation"].split()[0].lower() in sentence.lower():
                        q["reference"] = sentence.strip()
                        break
                valid.append(q)
            if len(valid) >= limit:
                break
        return valid
