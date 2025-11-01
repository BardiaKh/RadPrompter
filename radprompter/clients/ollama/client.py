from ..universal.client import UniversalClient

class OllamaClient(UniversalClient):
    """
    Ollama client using LiteLLM.
    
    Connects to Ollama for running open-source models locally.
    The model name will be automatically prefixed with 'ollama_chat/'.
    
    Examples:
        # Local Ollama (default)
        client = OllamaClient(model="gemma3:4b")
        
        # Custom Ollama server
        client = OllamaClient(
            model="gemma3:4b",
            api_base="http://localhost:11434"
        )
        
        # Custom parameters
        client = OllamaClient(
            model="gemma3:4b",
            temperature=0.0,
            seed=42
        )
        
        # Remote Ollama server
        client = OllamaClient(
            model="gemma3:4b",
            api_base="http://your-ollama-server:11434"
        )
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the Ollama client.
        
        Args:
            model (str): Model name (will be prefixed with 'ollama_chat/')
            api_base (str): Base URL of the Ollama server (default: http://localhost:11434)
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Set default api_base for Ollama if not provided
        if not kwargs.get("api_base"):
            kwargs["api_base"] = "http://localhost:11434"
        
        # Set default api_key to EMPTY if not provided
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "EMPTY"
        
        # Prefix model name with 'ollama_chat/' if not already present
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "ollama_chat/")
        elif not model.startswith("ollama_chat/"):
            model = f"ollama_chat/{model}"
        
        super().__init__(model, **kwargs) 