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