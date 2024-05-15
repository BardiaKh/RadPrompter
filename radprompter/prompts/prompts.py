import re
import html
import pickle
import datetime
import tomllib
import hashlib
from copy import deepcopy

try: 
    from IPython.display import HTML, display
    from IPython import get_ipython
except ImportError:
    pass

class Prompt:
    def __init__(self, prompt_file, debug=False):
        self.debug = debug

        assert prompt_file.endswith(".toml"), "Prompt file should be a TOML file."
        self.prompt_file = prompt_file
        self.data, self.raw_data = self.load_toml(self.prompt_file)
        self.md5_hash = hashlib.md5(str(self.data).encode()).hexdigest()
        self.version = self.data["METADATA"]["version"]

        self.system_prompt = self.init_constructor_component("system")
        self.user_prompts = self.init_constructor_component("user")
        self.response_templates = self.init_constructor_component("response_templates")
        self.stop_tags = self.init_constructor_component("stop_tags")
        self.num_turns = len(self.user_prompts)
        
        self.schemas = self.get_schemas()

        if self.response_templates is None:
            self.response_templates = [""]*self.num_turns
            
        if self.stop_tags is None:
            self.stop_tags = [""]*self.num_turns        
        
        if self.debug:
            print(self.num_turns)
        
        assert len(self.user_prompts) == len(self.response_templates) == len(self.stop_tags), "Number of user prompts, response templates, and stop tags should be the same."
    
    def process_schema(self, schema):
        processed_schema = []
        for item in schema:
            assert "variable_name" in item, "Schema item should have a 'variable_name' key."
            hint = f"'{item['variable_name']}'\n"
            if item['type'] == "select":
                if item['show_options_in_hint']:
                    hint += "Here are your options and you can explicitly use one of these:\n  - " + "\n  - ".join(f"`{i}`" for i in item['options']) + "\n\n"

            hint += "Hint: " + item['hint']

            other_values = {k:v for k,v in item.items() if k not in ["variable_name", "type", "options", "hint", "show_options_in_hint"]}
            processed_schema.append({
                "variable_name": item['variable_name'],
                "type": item['type'],
                "options": item['options'] if item['type'] == "select" else "",
                "hint": hint,
                **other_values
            })

        return processed_schema
    
    def get_schemas(self):
        if "SCHEMAS" in self.data:
            schemas = self.process_schema(self.data["SCHEMAS"].values())
        else:
            schemas = []
        if len(schemas) == 0:
            schemas = [{
                "variable_name": "default",
                "type": "default",
            }]
        if self.debug:
            print("\ninit_schema")
            print(schemas)
            
        return Schemas(self, schemas)
            
    def process_rdp(self, string):
        def replace_rdp(match):
            expression = match.group(1).strip()
            if '+' in expression:
                # Handle concatenation of prompts
                prompt_names = [name.strip() for name in expression.split('+')]
                prompts = [self.data['PROMPTS'].get(name) for name in prompt_names]
                value = ''.join(prompts)
            else:
                # Handle single prompt
                variable_name = expression
                if hasattr(self, variable_name):
                    value = getattr(self, variable_name)
                else:
                    value = self.data['PROMPTS'].get(variable_name)
            return value

        rdp_pattern = r'rdp\((.+?)\)'
        processed_string = re.sub(rdp_pattern, replace_rdp, string)
        return processed_string
    
    def init_constructor_component(self, component):
        if component not in self.data["CONSTRUCTOR"] and component not in ["system", "user"]:
            return None
                    
        component_data = self.data["CONSTRUCTOR"][component]
        if component == "system":
            component_data = self.process_rdp(component_data)
        else:
            if isinstance(component_data, str):
                component_data = [component_data]

            component_data = [self.process_rdp(i) for i in component_data]
        
        
        if self.debug:
            print(f"\n{component.title()}:")
            print(component_data)
            print("----------------")
            
        return component_data
                
    def load_toml(self, path):
        with open(path, "rb") as f:
            data = tomllib.load(f)

        with open(path, "r") as f:
            raw_data = f.read()
            
        return data, raw_data

    def replace_placeholders(self, item):
        for key in item:
            if "{{"+key+"}}" in self.system_prompt:
                self.system_prompt = self.system_prompt.replace("{{"+key+"}}", item[key])
            
            for i in range(self.num_turns):
                if "{{"+key+"}}" in self.user_prompts[i]:
                    self.user_prompts[i] = self.user_prompts[i].replace("{{"+key+"}}", item[key])

                if "{{"+key+"}}" in self.response_templates[i]:
                    self.response_templates[i] = self.response_templates[i].replace("{{"+key+"}}", item[key])
                                                             
    def verify_no_placeholders(self):
        remaining_placeholders = re.findall(r"{{(.*?)}}", self.system_prompt)
        if remaining_placeholders:
            raise ValueError(f"Unresolved placeholders: {remaining_placeholders}")
        
        for i in range(self.num_turns):
            remaining_placeholders = re.findall(r"{{(.*?)}}", self.user_prompts[i])
            if remaining_placeholders:
                raise ValueError(f"Unresolved placeholders: {remaining_placeholders}")
            
            remaining_placeholders = re.findall(r"{{(.*?)}}", self.response_templates[i])
            if remaining_placeholders:
                raise ValueError(f"Unresolved placeholders: {remaining_placeholders}")

        return True
        
    def __getitem__(self, index):
        schema = self.schemas[index]
        prompt_copy = deepcopy(self)
        prompt_copy.replace_placeholders(schema)
        return prompt_copy

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        try:
            # Check if running in an IPython notebook
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                # Return HTML representation for Jupyter notebook
                html_rep = "<div style='padding: 0; border-radius: 5px; font-family: Arial; line-height: 1.2rem; border: 1px solid currentColor'>"

                # System prompt
                html_rep += "<div style='display: flex; align-items: top; padding: 0; border-right-width: 1px'>"
                html_rep += "<h4 style='margin: 0; padding: 8px; flex: 0 0 100px; '>System:</h4>"
                html_rep += "<p style='margin: 0; padding: 8px; border-left: 1px solid currentColor;'>" + self._beautify_html(self.system_prompt) + "</p>"
                html_rep += "</div>"

                # User and assistant prompts
                for i in range(self.num_turns):
                    # User prompt
                    html_rep += "<div style='display: flex; align-items: top; padding: 0;'>"
                    html_rep += "<h4 style='margin: 0; padding: 8px; flex: 0 0 100px;'>User:</h4>"
                    html_rep += "<p style='margin: 0; padding: 8px; flex-grow: 1; border-left: 1px solid currentColor;border-top: 1px solid currentColor;'>" + self._beautify_html(self.user_prompts[i]) + "</p>"
                    html_rep += "</div>"

                    # Assistant prompt
                    html_rep += "<div style='display: flex; align-items: top; padding: 0;'>"
                    html_rep += "<h4 style='margin: 0; padding: 8px; flex: 0 0 100px;'>Assistant:</h4>"
                    html_rep += "<p style='margin: 0; padding: 8px; flex-grow: 1; border-left: 1px solid currentColor;border-top: 1px solid currentColor;'>" + self._beautify_html(self.response_templates[i])
                    html_rep += "<span style='background-color: rgb(178, 219, 255, 0.3);'>[... response ...]</span>"
                    html_rep += self._beautify_html(self.stop_tags[i]) + "</p>"
                    html_rep += "</div>"

                html_rep += "</div>"

                display(HTML(html_rep))
                return ""
            else:
                raise NameError

        except NameError:
            string_rep = "## SYSTEM:\n"
            string_rep += self.system_prompt + "\n\n"
            for i in range(self.num_turns):
                string_rep += "## USER:\n"
                string_rep += self.user_prompts[i] + "\n\n"
                string_rep += "## ASSISTANT:\n"
                string_rep +=  self.response_templates[i]+ "[... response ...]" + self.stop_tags[i] + "\n\n"
                
            return string_rep
    
    def _beautify_html(self, text):
        text = html.escape(text)
        text = text.replace("\n", "<br>")
        text = self._highlight_placeholders(text)
        return text
    
    def _highlight_placeholders(self, text):
        return text.replace("{{", "<span style='background-color: rgb(255, 224, 178, 0.3);'>{{").replace("}}", "}}</span>")
    
    
class Schemas:
    def __init__(self, prompt, schemas):
        self.prompt = prompt
        self.schemas = schemas

    def __getitem__(self, index):
        schema = self.schemas[index]
        prompt_copy = deepcopy(self.prompt)
        prompt_copy.replace_placeholders(schema)
        return prompt_copy

    def __len__(self):
        return len(self.schemas)