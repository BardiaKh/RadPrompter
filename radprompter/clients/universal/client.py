from ..client import Client
import warnings
import litellm # type: ignore
litellm.set_verbose = False

class UniversalClient(Client):
    """
    Universal client using LiteLLM that supports multiple LLM providers.
    
    Supported providers include:
    - OpenAI (gpt-4o, o3-pro, o4-mini, etc.)
    - Anthropic (claude 4 opus, claude 4 sonnet, claude 3.7 sonnet, etc.)
    - Google (gemini 2.5 pro, gemini 2.5 flash, gemini 2.0 flash, etc.)
    - Hugging Face (via inference endpoints)
    - vLLM
    - Ollama
    - And many more...
    
    Examples:
        # OpenAI
        client = UniversalClient(model="gpt-4")
        
        # Anthropic
        client = UniversalClient(model="claude-3-opus-20240229")
        
        # Ollama
        client = UniversalClient(
            model="ollama_chat/llama2",
            api_base="http://localhost:11434"
        )
        
        # vLLM
        client = UniversalClient(
            model="vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://localhost:8000"
        )
        
        # Custom deployment
        client = UniversalClient(
            model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="https://your-endpoint.com"
        )
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the UniversalClient.
        
        Args:
            model (str): Model name. Can include provider prefix (e.g., "openai/gpt-4", "anthropic/claude-3")
                        or use default routing (e.g., "gpt-4", "claude-3-opus-20240229")
            **kwargs: Additional parameters including:
                - api_key (str): API key for the provider
                - api_base (str): Base URL for the API (for custom deployments, vLLM, Ollama, etc.)
                - base_url (str): same as api_base for backwards compatibility
                - temperature (float): Sampling temperature (default: 0.0)
                - top_p (float): Top-p sampling (default: 1.0)
                - seed (int): Random seed for reproducibility
                - frequency_penalty (float): Frequency penalty (default: 0.0)
                - presence_penalty (float): Presence penalty (default: 0.0)
        """
        
        # Extract parameters
        self.api_key = kwargs.pop("api_key", None)
        self.api_base = kwargs.pop("api_base", None)
        base_url = kwargs.pop("base_url", None)
        if base_url:
            warnings.warn(
                "The 'base_url' parameter is deprecated and will be removed in version 3.0. "
                "Please use 'api_base' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.api_base = base_url
        
        self.temperature = kwargs.pop("temperature", 0.0)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.seed = kwargs.pop("seed", None)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.pop("presence_penalty", 0.0)
                
        # Ensure api_base does not end with /chat/completions or trailing /
        if self.api_base:
            self.api_base = self.api_base.rstrip('/')
            if self.api_base.endswith("/chat/completions"):
                self.api_base = self.api_base[:-len("/chat/completions")]

        self.provider = kwargs.pop("custom_llm_provider", None)
        if self.provider:
            assert self.api_base and self.api_key, "Please pass `api_base` and `api_key` to the client."
        
        # Get the provider from the model name
        if not self.provider:
            try:
                # Handling generic openai endpoints
                if len(model.split("/")) == 3 and model.split("/")[0] == "openai":
                    self.provider = model.split("/")[1]
                    assert self.api_base and self.api_key, "Please pass `api_base` and `api_key` to the client."
                else:
                    self.provider = litellm.get_llm_provider(model)[1]
            except:
                raise ValueError(f"Invalid model name: {model}. Please pass `custom_llm_provider` to the client.")
        
        # Store any additional kwargs for provider-specific needs
        self.extra_kwargs = kwargs

        super().__init__(model)

    def chat_complete(self, messages, stop=None, max_tokens=None, response_format=None):
        """
        Complete a chat conversation using LiteLLM.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            stop (str or list): Stop sequence(s) to end generation
            max_tokens (int): Maximum tokens to generate (overrides instance default)
            response_format (dict): Response format specification (e.g., JSON schema)
            
        Returns:
            str: The generated response text
        """
        # Prepare completion arguments
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": False,
        }
        
        # Add optional parameters if they exist
        if self.api_key:
            completion_args["api_key"] = self.api_key
        
        if self.api_base:
            completion_args["api_base"] = self.api_base
                
        if self.seed is not None:
            completion_args["seed"] = self.seed
        
        # Handle stop sequences
        if max_tokens:
            completion_args["max_tokens"] = max_tokens
        
        if stop:
            if isinstance(stop, str):
                completion_args["stop"] = [stop]
            else:
                completion_args["stop"] = stop
        
        # Handle response format (for structured output)
        if response_format:
            completion_args["response_format"] = response_format
        
        # Add any extra kwargs that might be needed for specific providers
        completion_args.update(self.extra_kwargs)
        
        try:
            # Make the completion request
            response = litellm.completion(**completion_args)
            
            # Extract the response text
            return response.choices[0].message.content
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"LiteLLM completion failed for model {self.model}: {str(e)}") from e
