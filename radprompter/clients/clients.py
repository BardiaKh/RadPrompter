from openai import OpenAI

class Client():
    def __init__(self, model):
        self.model = model
        
    def chat_complete(self, messages, stop, **kwargs):
        raise NotImplementedError()
    
    def ask_model(self, messages, stop, max_tokens=200):
        response = self.chat_complete(messages, stop, max_tokens)
        messages = self.update_last_message(messages, response, suffix=stop)
        return response, messages
        
    def update_last_message(self, messages, response, suffix=None):
        messages[-1]['content'] += response + (suffix if suffix else "")
        return messages
        
class OpenAIClient(Client):
    def __init__(self, model, **kwargs):
        Client.__init__(self, model)
        self.client = OpenAI(**kwargs)
        
    def chat_complete(self, messages, stop, max_tokens, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            frequency_penalty=1,
            stream=False,
            seed=42,
            temperature=0.0,
            stop=stop,
            logprobs=False,
            top_logprobs=0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content