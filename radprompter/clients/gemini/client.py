from ..universal.client import UniversalClient

class GeminiClient(UniversalClient):
    """
    Google Gemini client using LiteLLM.
    
    Supports all Google Gemini models including:
    - Gemini 2.5 Pro, Gemini 2.5 Flash
    - Gemini 1.5 Pro, Gemini 1.5 Flash
    - Gemini 1.0 Pro, Gemini 1.0 Pro Vision
    - Gemma models
    
    The model name will be automatically prefixed with 'gemini/' if not already present.
    
    Examples:
        # Standard Gemini
        client = GeminiClient(model="gemini-2.5-pro-preview-06-05")
        
        # With API key
        client = GeminiClient(
            model="gemini-2.5-pro-preview-06-05",
            api_key="your-gemini-api-key",
        )
        
        # Custom parameters
        client = GeminiClient(
            model="gemini-2.5-pro-preview-06-05",
            temperature=0.0,
            seed=42,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )        
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the Gemini client.
        
        Args:
            model (str): Gemini model name (will be prefixed with 'gemini/' if not already present)
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Prefix model name with 'gemini/' if not already present
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        
        super().__init__(model, **kwargs) 