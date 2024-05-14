import os
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class RadPrompter():
    def __init__(self, client, prompt, hide_blocks=False, output_directory=None, concurrency=1):
        self.client = client
        self.prompt = prompt
        self.hide_blocks = hide_blocks
        self.concurrency = concurrency
        self.log = {
            "Model": self.client.model,
            "Prompt TOML": self.prompt.prompt_file,
            "Prompt Version": self.prompt.version,
            "Prompt Hash": self.prompt.md5_hash,
            "Concurrency Factor": self.concurrency,
        }
        self.output_directory = output_directory
        
    def process_single_item(self, item, index):
        prompt = deepcopy(self.prompt)
        prompt.replace_placeholders(item)
            
        messages = [
            {"role": "system", "content": prompt.system_prompt},
        ]
        item_response = []
        for schema in prompt.schemas:
            schema_response = []
            prompt_with_schema = deepcopy(prompt)
            prompt_with_schema.replace_placeholders(schema)
        
            for i in range(prompt.num_turns):
                messages.append({"role": "user", "content": prompt_with_schema.user_prompts[i]})
                if prompt.response_templates[i] != "":
                    messages.append({"role": "assistant", "content": prompt_with_schema.response_templates[i]})
                
                response, messages = self.client.ask_model(messages, prompt_with_schema.stop_tags[i])
                schema_response.append(response)
            
            item_response.append({f"{schema['variable_name']}":schema_response})
            
            if self.hide_blocks:
                messages = [
                    {"role": "system", "content": prompt.system_prompt},
                ]
        
        return index, item_response

    def __call__(self, items):
        if not isinstance(items, list):
            items = [items]

        self.log['Start Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for index, item in enumerate(items):
                future = executor.submit(self.process_single_item, item, index)
                futures.append(future)

            results = []
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing items")):
                index, result = future.result()
                result.insert(0, {"index": index})
                result_keys = [list(r.keys())[0] for r in result]
                file_name = f"{index}.txt"
                for key, value in items[index].items():
                    if key not in result_keys:
                        if key == "filename":
                            file_name = f"{value}.txt"
                        else:
                            result.append({key: value})
                
                results.append(result)
                
                # save interim results
                if self.output_directory is not None:
                    if not os.path.exists(self.output_directory):
                        os.makedirs(self.output_directory)
                    output_dir = os.path.join(self.output_directory, file_name)
                    with open(output_dir, "w") as f:
                        for item in result:
                            f.write(str(item))
                            f.write("\n")

        self.log['End Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log['Duration'] = (datetime.strptime(self.log['End Time'], '%Y-%m-%d %H:%M:%S') - 
                                datetime.strptime(self.log['Start Time'], '%Y-%m-%d %H:%M:%S')).total_seconds()
        self.log['Number of Items'] = len(items)
        self.log['Average Processing Time'] = self.log['Duration'] / self.log['Number of Items']                
        return results
    
    def save_log(self, log_dir="./RadPrompter.log"):
        with open(log_dir, "w") as f:
            for key, value in self.log.items():
                f.write(f"{key}: {value}\n")
                
            f.write("\n\n")
            f.write("-"*20)
            f.write(" *** - Prompt Content - *** ")
            f.write("-"*20)
            f.write("\n")
            f.write(self.prompt.raw_data)
            
            f.close()