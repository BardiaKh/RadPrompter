from ..universal.client import UniversalClient

class vLLMClient(UniversalClient):
    """
    vLLM client using LiteLLM.
    
    Connects to vLLM inference servers for high-performance model serving.
    The model name will be automatically prefixed with 'hosted_vllm/'.
    
    Examples:
        # Local vLLM server
        client = vLLMClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://localhost:8000"
        )
        
        # Remote vLLM server
        client = vLLMClient(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            api_base="http://your-vllm-server:8000"
        )
        
        # Custom parameters
        client = vLLMClient(
            model="codellama/CodeLlama-7b-Instruct-hf",
            api_base="http://localhost:8000",
            temperature=0.0,
            seed=42
        )
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the vLLM client.
        
        Args:
            model (str): Model name (will be prefixed with 'hosted_vllm/')
            api_base (str): Base URL of the vLLM server (required)
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Ensure api_base is provided for vLLM
        if not kwargs.get("api_base") and not kwargs.get("base_url"):
            raise ValueError("api_base is required for vLLM client. Please provide the vLLM server URL.")
        
        # Set defalt api_key to EMPTY if not provided
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "EMPTY"
        
        # Prefix model name with 'hosted_vllm/' if not already present
        if not model.startswith("hosted_vllm/"):
            model = f"hosted_vllm/{model}"
        
        super().__init__(model, **kwargs)
    
    def chat_complete(self, messages, stop=None, max_tokens=None, response_format=None):
        """
        Complete a chat conversation using vLLM with automatic handling of vLLM-specific parameters.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            stop (str or list): Stop sequence(s) to end generation
            max_tokens (int): Maximum tokens to generate (overrides instance default)
            response_format (dict): Response format specification (e.g., JSON schema)
            **kwargs: Additional parameters
            
        Returns:
            str: The generated response text
        """
        # Automatically determine vLLM-specific parameters based on message structure
        vllm_params = {}
        
        # If the last message is an assistant message, we need to continue it
        # This happens when there's a response template/prefix
        if messages and messages[-1]['role'] == "assistant":
            vllm_params['continue_final_message'] = True
            vllm_params['add_generation_prompt'] = False
        
        # Pass all parameters to the parent method
        return super().chat_complete(
            messages=messages, 
            stop=stop, 
            max_tokens=max_tokens, 
            response_format=response_format, 
            **vllm_params,
        )