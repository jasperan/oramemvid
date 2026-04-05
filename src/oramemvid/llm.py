import json
from abc import ABC, abstractmethod
import httpx
from oramemvid.config import Settings

EXTRACTION_PROMPT = """Extract structured memory cards from the following text.
Return a JSON array of objects with these fields:
- entity: the subject (person, thing, concept)
- slot: the relationship or attribute
- value: the specific information
- kind: one of Fact, Preference, Event, Profile, Relationship, Goal
- confidence: 0.0 to 1.0

Return ONLY the JSON array, no other text. If no memories can be extracted, return [].

Text:
{content}"""


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...
    @abstractmethod
    def extract_memories(self, content: str) -> list[dict]: ...


class OllamaLLM(LLMProvider):
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def extract_memories(self, content: str) -> list[dict]:
        prompt = EXTRACTION_PROMPT.format(content=content)
        raw = self.complete(prompt)
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            cards = json.loads(cleaned)
            if not isinstance(cards, list):
                return []
            return cards
        except (json.JSONDecodeError, IndexError):
            return []


def get_llm_provider(settings: Settings) -> LLMProvider:
    return OllamaLLM(ollama_url=settings.ollama_url, model=settings.ollama_model)
