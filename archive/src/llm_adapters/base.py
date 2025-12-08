from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9) -> str:
        ...
