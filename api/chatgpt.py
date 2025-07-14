import os

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

if USE_GROQ:
    from groq import Groq
else:
    import openai

from api.prompt import Prompt

class ChatGPT:
    def __init__(self):
        self.prompt = Prompt()
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))
        self.frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY", 0))
        self.presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY", 0.6))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", 240))
        if USE_GROQ:
            self.api_key = os.getenv("GROQ_API_KEY")
            self.model = os.getenv("GROQ_MODEL", "gemma-7b-it")
            self.client = Groq(api_key=self.api_key)
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = os.getenv("OPENAI_MODEL", "text-davinci-003")

    def get_response(self):
        prompt_text = self.prompt.generate_prompt()
        if USE_GROQ:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt_text,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens
            )
            return response['choices'][0]['text'].strip()

    def add_msg(self, text):
        self.prompt.add_msg(text)
