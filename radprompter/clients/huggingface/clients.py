from ..clients import Client
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    import torch
    os.environ['HAS_TRANSFORMERS'] = str(True)
except ImportError:
    from abc import ABC
    StoppingCriteria = ABC
    os.environ['HAS_TRANSFORMERS'] = str(False)

class StopStringCriteria(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated tokens match the stop string tokens
        if input_ids.shape[1] < len(self.stop_ids):
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for stop_id in self.stop_ids:
            if stop_id not in input_ids[0, -len(self.stop_ids):]:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

class HuggingFaceClient(Client):
    def __init__(self, model_name, hf_model, hf_tokenizer, **kwargs):
        if os.environ['HAS_TRANSFORMERS'] != "True":
            raise ImportError("HuggingFaceClient requires the `transformers` package to be installed.")
        
        super().__init__(model_name)
        self.tokenizer = hf_tokenizer
        self.model = hf_model
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 0.0)
        self.max_tokens = kwargs.pop("max_tokens", 200)

    def chat_complete(self, messages, stop=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_tokens

        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )

        stopping_criteria_list = StoppingCriteriaList()
        if stop:
            stopping_criteria_list.append(StopStringCriteria(stop, self.tokenizer))

        outputs = self.model.generate(
            tokenized_chat['input_ids'], 
            max_new_tokens=max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            num_return_sequences=1,
            stopping_criteria=stopping_criteria_list
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response