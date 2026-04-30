import requests
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


class LocalLLMError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            raise LocalLLMError(f"Ollama call failed: {exc}") from exc
        data = resp.json()
        return data.get("message", {}).get("content", "")
