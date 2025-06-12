class Client():
    def __init__(self, model):
        self.model = model
        
    def chat_complete(self, messages, stop, max_tokens=None, response_format=None, **kwargs):
        raise NotImplementedError()
    
    def ask_model(self, messages, stop, max_tokens=200, response_format=None):
        if messages[-1]['role'] == "assistant":
            prefix = messages[-1]['content']
        else:
            prefix = ""
        response = self.chat_complete(messages, stop, max_tokens, response_format=response_format)
        messages = self.update_last_message(messages, response, prefix=prefix, suffix=stop)
        return response, messages
        
    def update_last_message(self, messages, response, prefix=None, suffix=None):
        if messages[-1]['role'] == "assistant":
            messages[-1]['content'] += (prefix if prefix else "") + response + (suffix if suffix else "")
        else:
            # If the last message is not an assistant message, add a new assistant message (there is no prefix)
            messages.append({"role": "assistant", "content": response + (suffix if suffix else "")})
        return messages