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
    def __init__(self, hf_model, hf_tokenizer, **kwargs):
        if os.environ['HAS_TRANSFORMERS'] != "True":
            raise ImportError("HuggingFaceClient requires the `transformers` package to be installed.")

        model_name = hf_model.__class__.__name__
        super().__init__(model_name)
        self.hf_tokenizer = hf_tokenizer
        self.hf_model = hf_model
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 0.0)
        self.max_tokens = kwargs.pop("max_tokens", 200)

    def chat_complete(self, messages, stop=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_tokens

        if messages[-1]['role'] == "assistant":
            last_message = messages[-1]['content']
            messages = messages[:-1]
        else:
            last_message = None
            
        tokenized_chat = self.hf_tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        if last_message:
            tokenized_last_message = self.hf_tokenizer.encode(last_message, return_tensors="pt", add_special_tokens=False)
            tokenized_chat = torch.cat([tokenized_chat, tokenized_last_message], dim=-1)

        stopping_criteria_list = StoppingCriteriaList()
        if stop:
            stopping_criteria_list.append(StopStringCriteria(stop, self.hf_tokenizer))

        outputs = self.hf_model.generate(
            tokenized_chat, 
            max_new_tokens=max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            num_return_sequences=1,
            stopping_criteria=stopping_criteria_list
        )

        prompt_size = tokenized_chat.size(-1)
        answer_tokens = outputs[0, prompt_size:]
        answer_text = self.hf_tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer_text