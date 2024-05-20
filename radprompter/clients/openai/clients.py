from openai import OpenAI
from .. import Client

class OpenAIClient(Client):
    def __init__(self, model, **kwargs):
        Client.__init__(self, model)
        self.seed = kwargs.pop("seed", 42)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 1)
        self.top_p = kwargs.pop("top_p", 1)
        self.client = OpenAI(**kwargs)

    def chat_complete(self, messages, stop, max_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            seed=self.seed,
            temperature=self.temperature,
            stop=stop if stop!="" else None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    
class vLLMClient(Client):
    def __init__(self, model, **kwargs):
        Client.__init__(self, model)
        if "api_key" not in kwargs:
            kwargs['api_key'] = "EMPTY"
        
        if "base_url" not in kwargs:
            raise ValueError("base_url must be provided for vLLMClient")
        self.seed = kwargs.pop("seed", 42)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 1)
        self.top_p = kwargs.pop("top_p", 1)
        self.client = OpenAI(**kwargs)

    def chat_complete(self, messages, stop, max_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            seed=self.seed,
            temperature=self.temperature,
            stop=stop if stop!="" else None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    
class OllamaClient(Client):
    def __init__(self, model, **kwargs):
        Client.__init__(self, model)
        if "api_key" not in kwargs:
            kwargs['api_key'] = "EMPTY"
        
        if "base_url" not in kwargs:
            raise ValueError("base_url must be provided for OllamaClient")
        self.seed = kwargs.pop("seed", 42)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 1)
        self.top_p = kwargs.pop("top_p", 1)
        self.client = OpenAI(**kwargs)

    def chat_complete(self, messages, stop, max_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            seed=self.seed,
            temperature=self.temperature,
            stop=stop if stop!="" else None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content