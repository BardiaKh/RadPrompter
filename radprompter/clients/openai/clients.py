from openai import OpenAI
from .. import Client

class OpenAIClient(Client):
    def __init__(self, model, **kwargs):
        self.seed = kwargs.pop("seed", 42)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 1)
        self.top_p = kwargs.pop("top_p", 1)
        self.client = OpenAI(**kwargs)
        super().__init__(model)

    def chat_complete(self, messages, stop, max_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            seed=self.seed,
            temperature=self.temperature,
            stop=stop if stop else None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    
class vLLMClient(OpenAIClient):
    def __init__(self, model, **kwargs):
        if "api_key" not in kwargs:
            kwargs['api_key'] = "EMPTY"
        
        if "base_url" not in kwargs:
            raise ValueError("base_url must be provided for vLLMClient")
        
        # Ensure base_url doesn't have trailing slash and ends with /v1
        base_url = kwargs['base_url'].rstrip('/')
        if not base_url.endswith('/v1'):
            base_url += '/v1'
        kwargs['base_url'] = base_url
        
        super().__init__(model, **kwargs)

class OllamaClient(OpenAIClient):
    def __init__(self, model, **kwargs):
        if "api_key" not in kwargs:
            kwargs['api_key'] = "EMPTY"
        
        if "base_url" not in kwargs:
            raise ValueError("base_url must be provided for OllamaClient")
        
        # Ensure base_url doesn't have trailing slash and ends with /v1
        base_url = kwargs['base_url'].rstrip('/')
        if not base_url.endswith('/v1'):
            base_url += '/v1'
        kwargs['base_url'] = base_url
        
        super().__init__(model, **kwargs)