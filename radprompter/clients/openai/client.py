from ..universal.client import UniversalClient

class OpenAIClient(UniversalClient):
    """
    OpenAI client using LiteLLM.
    
    Supports all OpenAI models including:
    - GPT-4o, GPT-4o-mini
    - GPT-4, GPT-4-turbo
    - GPT-3.5-turbo
    - o1, o1-mini, o1-preview
    - And other OpenAI models
    
    Examples:
        # Standard OpenAI
        client = OpenAIClient(model="gpt-4.1")
        
        # With API key
        client = OpenAIClient(
            model="gpt-4.1",
            api_key="your-openai-api-key"
        )
        
        # Custom parameters
        client = OpenAIClient(
            model="o4-mini",
            temperature=0.7,
            seed=42,
            reasoning_effort="medium",
        )
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the OpenAI client.
        
        Args:
            model (str): OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Prefix model name with 'openai/' if not already present
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        
        super().__init__(model, **kwargs) 