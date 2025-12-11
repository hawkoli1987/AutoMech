"""
Tests for Step 1.3: LLM Client Abstraction

Verifies:
- LLM client initialization and model detection
- Text generation
- JSON extraction from responses
- Structured output with Pydantic models
- VLM client with image support

Note: Tests requiring a live vLLM server are marked with @pytest.mark.live
      and skipped if server is not available.
"""

import os
import pytest
from pathlib import Path
from pydantic import BaseModel, Field


class TestLLMClientInit:
    """Test LLM client initialization."""
    
    def test_import_client(self):
        """Test importing LLM client."""
        from src.utils.llm_client import LLMClient, VLMClient, get_llm_client, get_vlm_client
    
    def test_client_creation_with_explicit_params(self):
        """Test creating client with explicit parameters (no server needed)."""
        from src.utils.llm_client import LLMClient
        
        # Mock the model detection by providing explicit model
        client = LLMClient(
            api_base="http://fake-server:8000",
            api_key="dummy",
            model="test-model",  # Explicit model skips detection
            temperature=0.5,
            max_tokens=1024,
        )
        
        assert client.model == "test-model"
        assert client.temperature == 0.5
        assert client.max_tokens == 1024
    
    def test_api_base_normalization(self):
        """Test that API base URL is normalized to include /v1."""
        from src.utils.llm_client import LLMClient
        
        # Without /v1
        client = LLMClient(
            api_base="http://localhost:8001",
            model="test-model",
        )
        assert client.api_base.endswith("/v1")
        
        # Already with /v1
        client = LLMClient(
            api_base="http://localhost:8001/v1",
            model="test-model",
        )
        assert client.api_base.endswith("/v1")
        assert not client.api_base.endswith("/v1/v1")
    
    def test_env_var_override(self):
        """Test that environment variables override config."""
        from src.utils.llm_client import LLMClient
        
        os.environ["OPENAI_API_BASE"] = "http://env-server:9000"
        os.environ["OPENAI_MODEL"] = "env-model"
        
        try:
            client = LLMClient()
            assert "env-server" in client.api_base
            assert client.model == "env-model"
        finally:
            del os.environ["OPENAI_API_BASE"]
            del os.environ["OPENAI_MODEL"]


class TestJSONExtraction:
    """Test JSON extraction from responses."""
    
    @pytest.fixture
    def client(self):
        """Create a client for testing extraction methods."""
        from src.utils.llm_client import LLMClient
        return LLMClient(api_base="http://fake:8000", model="test")
    
    def test_extract_json_simple(self, client):
        """Test extracting simple JSON."""
        text = '{"name": "test", "value": 42}'
        result = client._extract_json(text)
        assert result == {"name": "test", "value": 42}
    
    def test_extract_json_from_markdown(self, client):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:

```json
{
    "score": 0.85,
    "feedback": "Good match"
}
```

That's the output.'''
        
        result = client._extract_json(text)
        assert result["score"] == 0.85
        assert result["feedback"] == "Good match"
    
    def test_extract_json_array(self, client):
        """Test extracting JSON array."""
        text = '[1, 2, 3, 4, 5]'
        result = client._extract_json(text)
        assert result == [1, 2, 3, 4, 5]
    
    def test_extract_json_nested(self, client):
        """Test extracting nested JSON."""
        text = '''```
{
    "outer": {
        "inner": [1, 2, 3]
    }
}
```'''
        result = client._extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]
    
    def test_extract_json_invalid_raises(self, client):
        """Test that invalid JSON raises ValueError."""
        text = "This is not JSON at all"
        with pytest.raises(ValueError):
            client._extract_json(text)


class TestQwen3Support:
    """Test Qwen3-specific features."""
    
    def test_qwen3_extra_body(self):
        """Test that Qwen3 models get extra_body with thinking disabled."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient(
            api_base="http://fake:8000",
            model="Qwen/Qwen3-8B",
            enable_thinking=False,
        )
        
        extra_body = client._build_extra_body()
        assert extra_body is not None
        assert extra_body["chat_template_kwargs"]["enable_thinking"] is False
    
    def test_qwen3_thinking_enabled(self):
        """Test that Qwen3 thinking can be enabled."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient(
            api_base="http://fake:8000",
            model="Qwen/Qwen3-8B",
            enable_thinking=True,
        )
        
        extra_body = client._build_extra_body()
        assert extra_body["chat_template_kwargs"]["enable_thinking"] is True
    
    def test_non_qwen_no_extra_body(self):
        """Test that non-Qwen models don't get extra_body."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient(
            api_base="http://fake:8000",
            model="meta-llama/Llama-3-8B",
        )
        
        extra_body = client._build_extra_body()
        assert extra_body is None


class TestVLMClient:
    """Test VLM client features."""
    
    def test_vlm_client_inherits_llm(self):
        """Test that VLMClient inherits from LLMClient."""
        from src.utils.llm_client import VLMClient, LLMClient
        
        client = VLMClient(api_base="http://fake:8000", model="test-vlm")
        assert isinstance(client, LLMClient)
    
    def test_vlm_default_params(self):
        """Test VLM has different defaults."""
        from src.utils.llm_client import VLMClient
        
        client = VLMClient(api_base="http://fake:8000", model="test-vlm")
        assert client.temperature == 0.3  # Lower for VLM
        assert client.max_tokens == 1024


class TestStructuredOutput:
    """Test structured output with Pydantic models."""
    
    def test_structured_output_schema_generation(self):
        """Test that schema is correctly generated for Pydantic model."""
        from src.utils.llm_client import LLMClient
        
        class TestModel(BaseModel):
            score: float = Field(..., description="Score from 0 to 1")
            feedback: str = Field(..., description="Feedback text")
        
        client = LLMClient(api_base="http://fake:8000", model="test")
        
        # Check schema generation
        schema = TestModel.model_json_schema()
        assert "score" in schema["properties"]
        assert "feedback" in schema["properties"]


# =============================================================================
# Live Tests (require running vLLM server)
# =============================================================================

def is_server_available():
    """Check if vLLM server is available."""
    import requests
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_server_available(), reason="vLLM server not available")
class TestLLMClientLive:
    """Live tests requiring a running vLLM server."""
    
    def test_model_detection(self):
        """Test auto-detecting model from server."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient()  # Should auto-detect model
        assert client.model is not None
        assert len(client.model) > 0
        print(f"Detected model: {client.model}")
    
    def test_simple_generation(self):
        """Test simple text generation."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient()
        response = client.generate(
            prompt="What is 2 + 2? Reply with just the number.",
            max_tokens=10,
        )
        
        assert response is not None
        assert "4" in response
    
    def test_json_generation(self):
        """Test JSON generation."""
        from src.utils.llm_client import LLMClient
        
        client = LLMClient()
        result = client.generate_json(
            prompt='Return a JSON object with fields "a": 1 and "b": 2. Only output JSON.',
            max_tokens=50,
        )
        
        assert isinstance(result, dict)
        assert result.get("a") == 1
        assert result.get("b") == 2
    
    def test_structured_output(self):
        """Test structured output with Pydantic model."""
        from src.utils.llm_client import LLMClient
        
        class MathResult(BaseModel):
            result: int = Field(..., description="The calculation result")
            explanation: str = Field(..., description="Brief explanation")
        
        client = LLMClient()
        response = client.generate_structured(
            prompt="Calculate 10 + 5 and explain briefly.",
            response_model=MathResult,
            max_tokens=100,
        )
        
        assert isinstance(response, MathResult)
        assert response.result == 15


def is_vlm_server_available():
    """Check if VLM server is available on port 8002."""
    import requests
    api_base = os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_vlm_server_available(), reason="VLM server not available")
class TestVLMClientLive:
    """Live tests for VLM client requiring running Qwen3-VL server."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Get a sample image from the LLM4CAD dataset."""
        import glob
        data_root = "/scratch/Projects/SPEC-SF-AISG/source_files/AutoMech/data/LLM4CAD"
        images = glob.glob(f"{data_root}/**/img/*.png", recursive=True)
        if not images:
            pytest.skip("No sample images found in LLM4CAD dataset")
        return images[0]
    
    def test_vlm_model_detection(self):
        """Test VLM auto-detects model from server."""
        from src.utils.llm_client import VLMClient
        
        client = VLMClient(
            api_base=os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
        )
        assert client.model is not None
        assert "VL" in client.model or "vl" in client.model.lower()
        print(f"Detected VLM model: {client.model}")
    
    def test_vlm_simple_image_description(self, sample_image_path):
        """Test VLM can describe an image."""
        from src.utils.llm_client import VLMClient
        
        client = VLMClient(
            api_base=os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
        )
        
        response = client.generate_with_image(
            prompt="What type of mechanical part is shown in this image? Answer in one sentence.",
            image_path=sample_image_path,
            max_tokens=100,
        )
        
        assert response is not None
        assert len(response) > 10
        print(f"VLM response: {response}")
    
    def test_vlm_json_with_image(self, sample_image_path):
        """Test VLM can return structured JSON for an image."""
        from src.utils.llm_client import VLMClient
        
        client = VLMClient(
            api_base=os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
        )
        
        result = client.generate_json_with_image(
            prompt=(
                "Analyze this mechanical part image. "
                "Return a JSON object with: "
                '{"part_type": "string", "description": "string", "quality_score": 0.0 to 1.0}. '
                "Only output valid JSON."
            ),
            image_path=sample_image_path,
            max_tokens=200,
        )
        
        assert isinstance(result, dict)
        assert "part_type" in result or "description" in result
        print(f"VLM JSON response: {result}")
    
    def test_vlm_cad_quality_evaluation(self, sample_image_path):
        """Test VLM for CAD quality evaluation (simulating VLM judge)."""
        from src.utils.llm_client import VLMClient
        
        class CADEvaluation(BaseModel):
            visual_quality: float = Field(..., ge=0.0, le=1.0, description="Visual quality score")
            geometry_correctness: float = Field(..., ge=0.0, le=1.0, description="Geometry correctness score")
            feedback: str = Field(..., description="Feedback on the CAD render")
        
        client = VLMClient(
            api_base=os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
        )
        
        # Since VLM might not support structured output directly, test JSON generation
        result = client.generate_json_with_image(
            prompt=(
                "Evaluate this CAD render of a mechanical part. "
                "Return a JSON object with: "
                '{"visual_quality": 0.0-1.0, "geometry_correctness": 0.0-1.0, "feedback": "string"}. '
                "Only output valid JSON."
            ),
            image_path=sample_image_path,
            max_tokens=300,
        )
        
        assert isinstance(result, dict)
        if "visual_quality" in result:
            assert 0 <= result["visual_quality"] <= 1
        print(f"CAD Evaluation: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

