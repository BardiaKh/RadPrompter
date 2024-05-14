import os
from copy import deepcopy
from tqdm import tqdm
import tomllib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class RadPrompter():
    def __init__(self, client, prompt, hide_blocks=False, concurrency=1, output_directory=None):
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
        
    def process_single_item(self, item):
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

        if (len(item_response) == 1 and prompt.schemas[0]['variable_name'] == "default"):
            item_response = item_response[0]['default']
        
        return item_response

    def __call__(self, items):
        if not isinstance(items, list):
            items = [items]

        self.log['Start Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            report_ids = []
            for item in items:
                report_ids.append(item['report_id'])
                future = executor.submit(self.process_single_item, item)
                futures.append(future)

            results = []
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing items")):
                result = future.result()
                result.insert(0, {"report_id": [report_ids[i]]})
                results.append(result)
                
                # save result
                if self.output_directory is not None:
                    if not os.path.exists(self.output_directory):
                        os.makedirs(self.output_directory)
                    output_dir = os.path.join(self.output_directory, f"{report_ids[i]}")
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