import json
from copy import deepcopy
from pydantic import BaseModel, Field, create_model
from typing import Literal, Union, Optional
from enum import Enum


class Schemas:
    def __init__(self, prompt, schemas):
        self.prompt = prompt
        self.schemas = schemas

    def create_pydantic_model_for_schema(self, schema):
        """Create a Pydantic model for a single schema"""
        variable_name = schema['variable_name']
        schema_type = schema['type']
        
        # Determine field type and constraints
        if schema_type == "select":
            # Create enum for select type with choices
            assert "options" in schema, "Schema of type 'select' should have an 'options' key."
            options = schema['options']
            enum_class = Enum(f"{variable_name.title()}Enum", {opt: opt for opt in options})
            field_type = enum_class
            field_info = Field(description=schema.get('hint', ""))
        elif schema_type == "int":
            field_type = int
            field_info = Field(description=schema.get('hint', ""))
        elif schema_type == "float":
            field_type = float
            field_info = Field(description=schema.get('hint', ""))
        elif schema_type == "string":
            field_type = str
            field_info = Field(description=schema.get('hint', ""))
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        # Create the dynamic model
        model_name = f"{variable_name.title()}Model"
        model_fields = {variable_name: (field_type, field_info)}
        
        pydantic_model = create_model(model_name, **model_fields)
        return pydantic_model

    def populate_pydantic_models(self):
        """Populate Pydantic models for all schemas"""
        for schema in self.schemas:
            if schema['type'] != "default":
                schema['pydantic_model'] = self.create_pydantic_model_for_schema(schema)
                if schema.get('show_options_in_hint', False):
                    schema_text = f"\n\nRespond with a JSON object following this schema: {schema['pydantic_model'].model_json_schema()}"
                    schema['hint'] += schema_text
    
    def get_pydantic_model(self, index):
        """Get the Pydantic model for a specific schema"""
        return self.schemas[index].get('pydantic_model')
    
    def should_process_schema(self, schema_index, previous_responses):
        """
        Determine if a schema should be processed based on its dependencies.
        
        Args:
            schema_index (int): Index of the schema to check
            previous_responses (dict): Dictionary of previous schema responses
            
        Returns:
            tuple: (should_process: bool, default_value: any)
        """
        if schema_index >= len(self.schemas):
            return False, None
            
        schema = self.schemas[schema_index]
        
        # If no depends_on field, always process
        if 'depends_on' not in schema:
            return True, None
            
        depends_on = schema['depends_on']
        dependency_schema = depends_on['schema']
        condition_value = depends_on['condition']
        default_value = depends_on.get('default_value', '')
        
        # Find all response keys that match the dependency schema
        # This handles both single-turn ({schema}_response) and multi-turn ({schema}_response_0, etc.)
        matching_responses = []
        dependency_key_prefix = f"{dependency_schema}_response"
        
        for key, value in previous_responses.items():
            if key.startswith(dependency_key_prefix):
                # Check if it's exactly the base key or has a valid suffix
                if key == dependency_key_prefix or (key.startswith(dependency_key_prefix + "_") and 
                                                    key[len(dependency_key_prefix + "_"):].isdigit()):
                    matching_responses.append(value)
        
        # If no matching responses found, dependency not yet processed
        if not matching_responses:
            return False, default_value
        
        # Check if any of the matching responses meet the condition
        for actual_value in matching_responses:
            # Handle different condition types
            if isinstance(condition_value, str):
                condition_met = actual_value == condition_value
            elif isinstance(condition_value, list):
                condition_met = actual_value in condition_value
            else:
                # Try direct comparison
                condition_met = actual_value == condition_value
                
            if condition_met:
                return True, None
        
        # No matching responses met the condition
        return False, default_value
    
    def get_dependency_order(self):
        """
        Get the order in which schemas should be processed based on dependencies.
        
        Returns:
            list: List of schema indices in dependency order
        """
        # Simple topological sort for dependencies
        independent_schemas = []
        dependent_schemas = []
        
        for i, schema in enumerate(self.schemas):
            if 'depends_on' not in schema:
                independent_schemas.append(i)
            else:
                dependent_schemas.append(i)
        
        # For now, process independent schemas first, then dependent ones
        # In a more complex implementation, we'd do proper topological sorting
        return independent_schemas + dependent_schemas
    
    def parse_response(self, response_json, schema_index):
        """
        Extract just the response value from a Pydantic JSON response.
        
        Args:
            response_json (str or dict): The JSON response from the model
            schema_index (int): Index of the schema that was used
            
        Returns:
            The extracted value without the variable name key
            
        Raises:
            ValueError: If JSON parsing fails or expected key is not found
            IndexError: If schema_index is out of range
        """
        if schema_index >= len(self.schemas):
            raise IndexError(f"Schema index {schema_index} out of range. Available schemas: {len(self.schemas)}")
        
        schema = self.schemas[schema_index]
        if schema.get('pydantic_model') is None:
            return response_json
        
        variable_name = schema['variable_name']
        
        # Handle string JSON input
        if isinstance(response_json, str):
            try:
                response_dict = json.loads(response_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {e}")
        elif isinstance(response_json, dict):
            response_dict = response_json
        else:
            raise ValueError(f"Expected string or dict, got {type(response_json)}")
        
        # Extract the value for the variable name
        if variable_name not in response_dict:
            raise ValueError(f"Expected key '{variable_name}' not found in response. Available keys: {list(response_dict.keys())}")
        
        return response_dict[variable_name]
    
    def __getitem__(self, index):
        schema = self.schemas[index]
        prompt_copy = deepcopy(self.prompt)
        prompt_copy.replace_placeholders(schema)
        return prompt_copy

    def __len__(self):
        return len(self.schemas) 