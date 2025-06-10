from ..clients import Client
import os
import anthropic

class AnthropicClient(Client):
    def __init__(self, model="claude-3-sonnet-20240229", **kwargs):
        super().__init__(model)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.max_tokens = kwargs.pop("max_tokens", 200)

    def chat_complete(self, messages, stop=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Extract system prompt if present
        system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)

        # Convert messages to Anthropic's format
        anthropic_messages = []
        for message in messages:
            if message['role'] == 'user':
                anthropic_messages.append(anthropic.Message(role="user", content=message['content']))
            elif message['role'] == 'assistant':
                anthropic_messages.append(anthropic.Message(role="assistant", content=message['content']))

        response = self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_sequences=stop if stop else None,
            system=system_prompt
        )

        return response.content[0].text


