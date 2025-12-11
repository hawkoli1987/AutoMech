"""
LLM Client Abstraction

OpenAI-compatible client for local vLLM servers and cloud APIs.
Follows patterns from inference/client/ for model flexibility.

Environment Variables:
    OPENAI_API_BASE: Base URL for API (e.g., http://localhost:8001)
    OPENAI_API_KEY: API key (can be 'dummy' for vLLM)
    OPENAI_MODEL: Optional model name (auto-detected if not set)
"""

import json
import os
import re
import time
from typing import Any, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.config import get_config, LLMConfig, VLMConfig


T = TypeVar('T', bound=BaseModel)


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """
    OpenAI-compatible LLM client for vLLM and cloud APIs.
    
    Features:
    - Auto-detects model from /v1/models endpoint
    - Supports structured output (JSON mode)
    - Retry logic with exponential backoff
    - Qwen3-specific parameters (disable thinking mode)
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
        enable_thinking: bool = False,
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_base: Base URL for the API. Uses OPENAI_API_BASE env or config.
            api_key: API key. Uses OPENAI_API_KEY env or config.
            model: Model name. Auto-detected from server if not provided.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            enable_thinking: Enable Qwen3 thinking mode (default: False).
        """
        config = get_config()
        llm_config = config.llm
        
        # Resolve API settings
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", llm_config.api_base)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", llm_config.api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        
        # Ensure base URL ends with /v1
        if not self.api_base.endswith("/v1"):
            self.api_base = self.api_base.rstrip("/") + "/v1"
        
        # Create OpenAI client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "dummy",
            timeout=self.timeout,
        )
        
        # Resolve model
        self._model = model or os.getenv("OPENAI_MODEL")
        if self._model is None:
            self._model = self._detect_model()
        
        # Default generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _detect_model(self) -> str:
        """Detect model ID from /v1/models endpoint."""
        try:
            models = self.client.models.list()
            if hasattr(models, "data") and len(models.data) > 0:
                return models.data[0].id
        except Exception as e:
            raise RuntimeError(f"Failed to detect model from server: {e}")
        raise RuntimeError("No models available on server")
    
    @property
    def model(self) -> str:
        """Get the model ID."""
        return self._model
    
    def _build_extra_body(self) -> Optional[dict]:
        """Build extra_body for Qwen3 specific parameters."""
        # Check if this is a Qwen3 model
        if "qwen3" in self._model.lower() or "qwen-3" in self._model.lower():
            return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            stop: Stop sequences.
        
        Returns:
            Generated text string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Chat completion with messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            stop: Stop sequences.
        
        Returns:
            Generated text string.
        """
        request_params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        
        if stop:
            request_params["stop"] = stop
        
        extra_body = self._build_extra_body()
        if extra_body:
            request_params["extra_body"] = extra_body
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict | list:
        """
        Generate JSON output from a prompt.
        
        Extracts JSON from the response, handling markdown code blocks.
        
        Args:
            prompt: User prompt requesting JSON output.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
        
        Returns:
            Parsed JSON as dict or list.
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else 0.3,  # Lower temp for JSON
            max_tokens=max_tokens,
        )
        
        return self._extract_json(response)
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Generate structured output matching a Pydantic model.
        
        Args:
            prompt: User prompt requesting structured output.
            response_model: Pydantic model class to validate output.
            system_prompt: Optional system prompt (schema info appended).
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
        
        Returns:
            Instance of response_model.
        """
        # Build enhanced system prompt with schema
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        enhanced_system = system_prompt or "You are a helpful assistant."
        enhanced_system += f"\n\nRespond with valid JSON matching this schema:\n```json\n{schema_str}\n```"
        
        json_response = self.generate_json(
            prompt=prompt,
            system_prompt=enhanced_system,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=max_tokens,
        )
        
        return response_model.model_validate(json_response)
    
    def _extract_json(self, text: str) -> dict | list:
        """
        Extract JSON from text, handling markdown code blocks.
        
        Args:
            text: Text potentially containing JSON.
        
        Returns:
            Parsed JSON object.
        """
        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
            r'(\{[\s\S]*\})',                 # { ... }
            r'(\[[\s\S]*\])',                 # [ ... ]
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to extract JSON from response: {e}\nResponse: {text[:500]}")


# =============================================================================
# VLM Client
# =============================================================================

class VLMClient(LLMClient):
    """
    Vision-Language Model client for multimodal inputs.
    
    Extends LLMClient to support image inputs in messages.
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize VLM client with VLM-specific defaults."""
        config = get_config()
        vlm_config = config.vlm
        
        super().__init__(
            api_base=api_base or vlm_config.api_base,
            api_key=api_key or vlm_config.api_key,
            model=model or vlm_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text with an image input.
        
        Args:
            prompt: Text prompt.
            image_path: Path to image file.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
        
        Returns:
            Generated text.
        """
        import base64
        from pathlib import Path
        
        # Read and encode image
        image_data = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine media type
        suffix = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(suffix, 'image/png')
        
        # Build message with image
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def generate_json_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict | list:
        """
        Generate JSON with an image input.
        
        Args:
            prompt: Text prompt requesting JSON output.
            image_path: Path to image file.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
        
        Returns:
            Parsed JSON.
        """
        response = self.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=max_tokens,
        )
        
        return self._extract_json(response)


# =============================================================================
# Factory Functions
# =============================================================================

def get_llm_client(**kwargs) -> LLMClient:
    """
    Get a configured LLM client instance.
    
    Uses configuration from config file and environment variables.
    Kwargs override defaults.
    """
    return LLMClient(**kwargs)


def get_vlm_client(**kwargs) -> VLMClient:
    """
    Get a configured VLM client instance.
    
    Uses configuration from config file and environment variables.
    Kwargs override defaults.
    """
    return VLMClient(**kwargs)

