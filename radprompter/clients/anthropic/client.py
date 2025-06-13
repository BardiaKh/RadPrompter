from ..universal.client import UniversalClient

class AnthropicClient(UniversalClient):
    """
    Anthropic client using LiteLLM.
    
    Supports all Anthropic Claude models including:
    - Claude 4 Opus, Claude 4 Sonnet
    - Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3.5 Haiku
    - Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
    - Claude 2.1, Claude 2.0
    - Claude Instant
    
    Examples:
        # Standard Anthropic
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        
        # With API key
        client = AnthropicClient(
            model="claude-sonnet-4-20250514",
            api_key="your-anthropic-api-key"
        )
        
        # Custom parameters
        client = AnthropicClient(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            seed=42
        )
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the Anthropic client.
        
        Args:
            model (str): Anthropic model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            **kwargs: Additional parameters passed to UniversalClient
        """
        # Prefix model name with 'anthropic/' if not already present
        if not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        
        super().__init__(model, **kwargs) 