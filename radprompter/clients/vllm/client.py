from ..universal.client import UniversalClient

class vLLMClient(UniversalClient):
    """
    vLLM client using LiteLLM.
    
    Connects to vLLM inference servers for high-performance model serving.
    The model name will be automatically prefixed with 'vllm/'.
    
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
            model (str): Model name (will be prefixed with 'vllm/')
            api_base (str): Base URL of the vLLM server (required)
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Ensure api_base is provided for vLLM
        if not kwargs.get("api_base") and not kwargs.get("base_url"):
            raise ValueError("api_base is required for vLLM client. Please provide the vLLM server URL.")
        
        # Set defalt api_key to EMPTY if not provided
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "EMPTY"
        
        # Prefix model name with 'vllm/' if not already present
        if not model.startswith("vllm/"):
            model = f"vllm/{model}"
        
        super().__init__(model, **kwargs) 