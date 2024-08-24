import os
import pandas as pd
import re
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from .clients import OpenAIClient, HuggingFaceClient
from .__version__ import __version__

class RadPrompter():
    def __init__(self, client, prompt, output_file, hide_blocks=False, concurrency=1, max_generation_tokens=4096):
        self.client = client
        self.prompt = prompt
        self.hide_blocks = hide_blocks
        self.concurrency = concurrency
        self.output_file = output_file
        self.max_generation_tokens = max_generation_tokens
        assert self.output_file.endswith(".csv"), "Output file must be a .csv file"
        file_exists = os.path.isfile(self.output_file)
        
        if file_exists:
            print(f"WARNING: Output file {self.output_file} already exists. The file will be **replaced** if you proceed with running the engine.")
        
        if type(self.client) == OpenAIClient and self.prompt.response_templates.count("") != prompt.num_turns:
            print("WARNING: OpenAI client does not accept response templates and will be ignored.")
            self.prompt.response_templates = [""]*prompt.num_turns
            
        if isinstance(self.client, HuggingFaceClient) and self.concurrency > 1:
            print("WARNING: HuggingFace client does not support concurrency > 1 and will be set to 1.")
            self.concurrency = 1
        
        self.log = {
            "RadPrompter Version": __version__,
            "Model": self.client.model,
            "Seed": self.client.seed,
            "Temperature": self.client.temperature,
            "Frequency Penalty": self.client.frequency_penalty,
            "Top-P": self.client.top_p,
            "Prompt TOML": self.prompt.prompt_file,
            "Prompt Version": self.prompt.version,
            "Prompt Hash": self.prompt.md5_hash,
            "Concurrency Factor": self.concurrency,
        }
        
    def process_single_item(self, item, index):
        prompt = deepcopy(self.prompt)
            
        messages = [
            {"role": "system", "content": prompt.system_prompt},
        ]
        item_response = []
        for schema in prompt.schemas.schemas:
            try:
                schema_response = []
                prompt_with_schema = deepcopy(prompt)
                merged_dict = deepcopy(schema)
                merged_dict.update(item)
                prompt_with_schema.replace_placeholders(merged_dict)
            
                for i in range(prompt.num_turns):
                    messages.append({"role": "user", "content": prompt_with_schema.user_prompts[i]})
                    if prompt.response_templates[i] != "":
                        messages.append({"role": "assistant", "content": prompt_with_schema.response_templates[i]})
                    
                    response, messages = self.client.ask_model(messages, prompt_with_schema.stop_tags[i], max_tokens=self.max_generation_tokens)
                    schema_response.append(response)
                                                    
                if len(schema_response) == 1:
                    item_response.append({f"{schema['variable_name']}_response":schema_response[0]})
                else:
                    for r, schema_response_ in enumerate(schema_response):    
                        item_response.append({f"{schema['variable_name']}_response_{r}":schema_response_})
                            
                if self.hide_blocks:
                    messages = [
                        {"role": "system", "content": prompt.system_prompt},
                    ]
            except Exception as e:
                print(f"Error processing schema {schema['variable_name']} for item {index}: {e}. You might need to increase engine's `max_generation_tokens` parameters.")
        
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

            if self.output_file is not None:
                with open(self.output_file, "w", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    header_written = False

                    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing items")):
                        index, result = future.result()
                        result.insert(0, {"index": index})
                        result_keys = [list(r.keys())[0] for r in result]
                        for key, value in items[index].items():
                            if key not in result_keys:
                                result.append({key: value})

                        if not header_written:
                            # Write header only if it hasn't been written yet
                            header = [key for r in result for key in r.keys()]
                            writer.writerow(header)
                            header_written = True

                        row = []
                        for r in result:
                            key = list(r.keys())[0]
                            value = r[key]
                            if isinstance(value, list):
                                value = "|".join(str(v) for v in value)
                            row.append(value)
                        writer.writerow(row)

        self.log['End Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log['Duration'] = (datetime.strptime(self.log['End Time'], '%Y-%m-%d %H:%M:%S') - 
                                datetime.strptime(self.log['Start Time'], '%Y-%m-%d %H:%M:%S')).total_seconds()
        self.log['Number of Items'] = len(items)
        self.log['Average Processing Time'] = self.log['Duration'] / self.log['Number of Items']                
    
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
            
    def sanitize_json(self, variable_name="all"):
        df = pd.read_csv(self.output_file, index_col='index')

        if variable_name == "all":
            for schema in self.prompt.schemas.schemas:
                if schema["type"] == "select":
                    column_name = f"{schema['variable_name']}_response"
                    options = schema["options"]
                    df[column_name] = df[column_name].apply(lambda x: self._sanitize_response(x, options))
        else:
            schema = next((s for s in self.prompt.schemas.schemas if s["variable_name"] == variable_name), None)
            if schema and schema["type"] == "select":
                column_name = f"{schema['variable_name']}_response"
                options = schema["options"]
                df[column_name] = df[column_name].apply(lambda x: self._sanitize_response(x, options))

        return df

    def _sanitize_response(self, response, options):
        matches = []
        for option in options:
            pattern = r'\b' + re.escape(option) + r'\b'
            if re.search(pattern, response, re.IGNORECASE):
                matches.append(option)

        if len(matches) == 1:
            return matches[0]
        else:
            return "**RECHECK** " + response
