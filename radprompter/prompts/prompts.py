import re
import html
import pickle
import datetime
import tomllib


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
        self.data = self.load_toml(self.prompt_file)
        self.version = self.data["METADATA"]["version"]

        self.init_system_prompt()
        self.init_user_prompts()
        self.init_response_templates()
        self.init_stop_tags()
        self.init_schema()
        
        self.num_turns = len(self.user_prompts)
        
        if self.debug:
            print(self.num_turns)
        
        assert len(self.user_prompts) == len(self.response_templates) == len(self.stop_tags), "Number of user prompts, response templates, and stop tags should be the same."
    
    def process_schema(self, schema):
        processed_schema = []
        for item in schema:
            hint = f"'{item['name']}'\n"
            if item['show_options_in_hint']:
                hint += "Here are your options and you can explicitly use one of these:\n  - " + "\n  - ".join(f"`{i}`" for i in item['options']) + "\n\n"

            hint += "Hint: " + item['hint']

            processed_schema.append({
                "variable": item['name'],
                "CoT": item['CoT'],
                "hint": hint,
                "type": item['type'],
                "options": item['options'] if item['type'] == "select" else None,
            })

        return processed_schema
    
    def init_schema(self):
        self.schema = self.process_schema(self.data["SCHEMA"].values())
        if self.debug:
            print("\ninit_schema")
            print(self.schema)
    
    def init_system_prompt(self):
        if self.data["CONSTRUCTOR"]["system"] in self.data["PROMPTS"].keys():
            self.system_prompt = self.data["PROMPTS"][self.data["CONSTRUCTOR"]["system"]]
        else:
            self.system_prompt = self.data["CONSTRUCTOR"]["system"]
        if self.debug:
            print("\ninit_system_prompt")
            print(self.data["CONSTRUCTOR"]["system"])
            print(self.system_prompt)
    
    def init_response_templates(self):
            
        response_templates = []
        for value in self.data["CONSTRUCTOR"]["response_templates"]:
            if value in self.data["PROMPTS"].keys():
                response_templates.append(self.data["PROMPTS"][value])
            else:
                response_templates.append(value)
        
        if self.debug:
            print("\ninit_response_templates")
            print(self.data["CONSTRUCTOR"]["response_templates"])
            print(response_templates)
            
        self.response_templates = response_templates

    def init_stop_tags(self):
        stop_tags = []
        for value in self.data["CONSTRUCTOR"]["stop_tags"]:
            if value in self.data["PROMPTS"].keys():
                stop_tags.append(self.data["PROMPTS"][value])
            else:
                stop_tags.append(value)
        
        if self.debug:
            print("\ninit_stop_tags")
            print(self.data["CONSTRUCTOR"]["stop_tags"])
            print(stop_tags)
            
        self.stop_tags = stop_tags

    def init_user_prompts(self):
        user_prompts = self.data["CONSTRUCTOR"]["user"]
        prompts = []
        for i in user_prompts:
            prompt = ""
            for j in [k.strip() for k in i.split("+")]:
                prompt += self.data["PROMPTS"][j]
            prompts.append(prompt)
        if self.debug:
            print("\ninit_user_prompts")
            print(user_prompts)
            print(prompts)
        self.user_prompts = prompts
        
    def load_toml(self, path):
        with open(path, "rb") as f:
            return tomllib.load(f)

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

    def save(self, path):
        path = path if path.endswith(".rdp") else path + ".rdp"
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path):
        # Load the object from the file
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    
    
if __name__=="__main__":
    prompt_file = "/home/tdapame/development/RadPrompter/demo_schema.toml"
    prompt = Prompt(prompt_file, debug=True)

    print(prompt.system_prompt)
    print(prompt.user_prompts)
    print(prompt.response_templates)
    print(prompt.stop_tags)
    
    print(prompt)
    