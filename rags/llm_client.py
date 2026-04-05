import requests
from openai import OpenAI
from types import SimpleNamespace

class LLMClient:
    def __init__(self, provider="ollama", model="llama3", base_url=None, api_key=None):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "ollama":
            self.base_url = base_url or "http://localhost:11434"

    class Chat:
        def __init__(self, outer):
            self.outer = outer

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, model, messages, **kwargs):
                outer = self.outer.outer

                if outer.provider == "openai":
                    return outer.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )

                elif outer.provider == "ollama":
                    prompt = self._messages_to_prompt(messages)
                    raw = requests.post(
                        f"{outer.base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    raw.raise_for_status()
                    data = raw.json()

                    if "response" not in data:
                        raise ValueError(f"Unexpected Ollama response: {data}")

                    return self._wrap_response(data["response"])

            def _messages_to_prompt(self, messages):
                prompt = ""
                for m in messages:
                    role = m["role"]
                    content = m["content"]
                    prompt += f"{role.upper()}: {content}\n"
                prompt += "ASSISTANT:"
                return prompt

            def _wrap_response(self, text):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                role="assistant",
                                content=text
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0
                    )
                )

        @property
        def completions(self):
            return LLMClient.Chat.Completions(self)

    @property
    def chat(self):
        return LLMClient.Chat(self)