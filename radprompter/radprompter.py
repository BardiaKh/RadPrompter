import os
import pandas as pd
import re
import warnings
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from .clients import HuggingFaceClient, OpenAIClient
from .__version__ import __version__

class RadPrompter():
    def __init__(self, client, prompt, output_file, hide_blocks=False, concurrency=1, max_generation_tokens=4096, use_pydantic=True):
        self.client = client
        self.prompt = prompt
        self.hide_blocks = hide_blocks
        self.concurrency = concurrency
        self.output_file = output_file
        self.max_generation_tokens = max_generation_tokens
        self.use_pydantic = use_pydantic
        assert self.output_file.endswith(".csv"), "Output file must be a .csv file"
        file_exists = os.path.isfile(self.output_file)
        
        if file_exists:
            warnings.warn(f"Output file {self.output_file} already exists. The file will be **replaced** if you proceed with running the engine.")
        
        if isinstance(self.client, OpenAIClient) and self.prompt.response_templates.count("") != prompt.num_turns:
            warnings.warn("OpenAI models do not accept response templates and will be ignored.")
            self.prompt.response_templates = [""]*prompt.num_turns
            
        if isinstance(self.client, HuggingFaceClient) and self.concurrency > 1:
            warnings.warn("HuggingFace client does not support concurrency > 1 and will be set to 1.")
            self.concurrency = 1
        
        if isinstance(self.client, HuggingFaceClient) and self.use_pydantic:
            warnings.warn("HuggingFace client does not support Pydantic models and will be set to False.")
            self.use_pydantic = False
        
        # Populate Pydantic models if use_pydantic is True
        if self.use_pydantic:
            if len(self.prompt.schemas.schemas) == 1 and self.prompt.schemas.schemas[0]['type']=="default":
                self.use_pydantic = False
            else:
                self.prompt.schemas.populate_pydantic_models()
        
        if self.use_pydantic:
            if self.prompt.stop_tags.count("") != self.prompt.num_turns or self.prompt.response_templates.count("") != self.prompt.num_turns:
                warnings.warn("Pydantic models do not support stop tags and response templates and will be ignored.")
                self.prompt.reset_stop_tags()
                self.prompt.reset_response_templates()
        
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
            "Use Pydantic": self.use_pydantic,
        }
        
    def process_single_item(self, item, index):
        prompt = deepcopy(self.prompt)
            
        messages = [
            {"role": "system", "content": prompt.system_prompt},
        ]
        item_response = []
        previous_responses = {}  # Store responses for dependency checking
        
        # Get schemas in dependency order
        schema_order = prompt.schemas.get_dependency_order()
        
        for schema_idx in schema_order:
            schema = prompt.schemas.schemas[schema_idx]
            try:
                # Check if this schema should be processed based on dependencies
                should_process, default_value = prompt.schemas.should_process_schema(schema_idx, previous_responses)
                
                if not should_process:
                    # Use default value for skipped schema
                    response_key = f"{schema['variable_name']}_response"
                    previous_responses[response_key] = default_value
                    item_response.append({response_key: default_value})
                    continue
                
                schema_response = []
                prompt_with_schema = deepcopy(prompt)
                merged_dict = deepcopy(schema)
                merged_dict.update(item)
                prompt_with_schema.replace_placeholders(merged_dict)
                
                additional_generation_params = {}
                
                # Get response format if using Pydantic
                response_format = None
                if self.use_pydantic and schema.get('pydantic_model'):
                    response_format = prompt.schemas.get_pydantic_model(schema_idx)
                                
                for i in range(prompt.num_turns):
                    messages.append({"role": "user", "content": prompt_with_schema.user_prompts[i]})
                    if prompt.response_templates[i] != "":
                        messages.append({"role": "assistant", "content": prompt_with_schema.response_templates[i]})
                    
                    response, messages = self.client.ask_model(
                        messages, 
                        prompt_with_schema.stop_tags[i], 
                        max_tokens=self.max_generation_tokens, 
                        response_format=response_format,
                        **additional_generation_params
                    )
                    
                    # Parse the response if using Pydantic
                    parsed_response = self.prompt.schemas.parse_response(response, schema_idx)
                    schema_response.append(parsed_response)
                                                                                        
                if len(schema_response) == 1:
                    response_key = f"{schema['variable_name']}_response"
                    response_value = schema_response[0]
                    previous_responses[response_key] = response_value
                    item_response.append({response_key: response_value})
                else:
                    for r, schema_response_ in enumerate(schema_response):    
                        response_key = f"{schema['variable_name']}_response_{r}"
                        previous_responses[response_key] = schema_response_
                        item_response.append({response_key: schema_response_})
                            
                if self.hide_blocks:
                    messages = [
                        {"role": "system", "content": prompt.system_prompt},
                    ]
            except Exception as e:
                print(f"Error processing schema {schema['variable_name']} for item {index}: {e}")
                # Add empty response for failed schema to maintain consistency
                response_key = f"{schema['variable_name']}_response"
                previous_responses[response_key] = ""
                item_response.append({response_key: "ERROR"})
        
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
        
        # Add log metadata as comments at the beginning of the CSV file
        if self.output_file is not None:
            self._add_metadata_to_csv()
    
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
            
    def _add_metadata_to_csv(self):
        with open(self.output_file, "r") as f:
            lines = f.readlines()
        
        with open(self.output_file, "w") as f:
            f.write(f"# RadPrompter Version: {self.log['RadPrompter Version']}\n")
            f.write(f"# Model: {self.log['Model']}\n")
            f.write(f"# Seed: {self.log['Seed']}\n")
            f.write(f"# Temperature: {self.log['Temperature']}\n")
            f.write(f"# Frequency Penalty: {self.log['Frequency Penalty']}\n")
            f.write(f"# Top-P: {self.log['Top-P']}\n")
            f.write(f"# Prompt TOML: {self.log['Prompt TOML']}\n")
            f.write(f"# Prompt Version: {self.log['Prompt Version']}\n")
            f.write(f"# Prompt Hash: {self.log['Prompt Hash']}\n")
            f.write(f"# Concurrency Factor: {self.log['Concurrency Factor']}\n")
            f.write(f"# Use Pydantic: {self.log['Use Pydantic']}\n")
            f.write(f"# Start Time: {self.log['Start Time']}\n")
            f.write(f"# End Time: {self.log['End Time']}\n")
            f.write(f"# Duration: {self.log['Duration']}\n")
            f.write(f"# Number of Items: {self.log['Number of Items']}\n")
            f.write(f"# Average Processing Time: {self.log['Average Processing Time']}\n")
            f.write("\n")