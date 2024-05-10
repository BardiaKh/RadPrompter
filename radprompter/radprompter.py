from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class RadPrompter():
    def __init__(self, client, prompt, concurrency=1):
        self.client = client
        self.prompt = prompt
        self.concurrency = concurrency
        
    def process_single_item(self, item):
        prompt = deepcopy(self.prompt)
        prompt.replace_placeholders(item)
        
        messages = [
            {"role": "system", "content": prompt.system_prompt},
        ]
        
        for i in range(prompt.num_turns):
            messages.append({"role": "user", "content": prompt.user_prompts[i]})
            messages.append({"role": "assistant", "content": prompt.response_templates[i]})
            
            response, messages = self.client.ask_model(messages, prompt.stop_tags[i])
            
            print(response)
            
        return messages

    def __call__(self, items):
        if not isinstance(items, list):
            items = [items]
            
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for item in items:
                future = executor.submit(self.process_single_item, item)
                futures.append(future)

            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
                result = future.result()
                results.append(result)
                
        return results