"""A package for simplified and reproducible LLM prompting."""

from .radprompter import RadPrompter
from .prompts import Prompt
from .clients import OpenAIClient, vLLMClient, OllamaClient

__version__ = "1.0.7"