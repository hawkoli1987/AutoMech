"""
Configuration management for the Agentic CAD Framework.

Loads configuration from YAML files with environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# =============================================================================
# Configuration Models
# =============================================================================

class LLMConfig(BaseModel):
    """LLM configuration."""
    api_base: str = "http://localhost:8001"
    api_key: str = "dummy"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_thinking: bool = False
    max_retries: int = 3
    timeout: int = 60


class VLMConfig(BaseModel):
    """VLM configuration."""
    api_base: str = "http://localhost:8001"
    api_key: str = "dummy"
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024


class AgentConfig(BaseModel):
    """Agent configuration."""
    param_score_threshold: float = 0.85
    vlm_score_threshold: float = 0.80
    max_iterations: int = 5
    enable_vlm_judge: bool = True
    enable_param_judge: bool = True


class DataConfig(BaseModel):
    """Data configuration."""
    llm4cad_path: str = "data/LLM4CAD"
    categories: Optional[list[str]] = None
    max_samples_per_category: Optional[int] = None
    shuffle: bool = False
    random_seed: int = 42


class CADConfig(BaseModel):
    """CAD generation configuration."""
    output_format: str = "step"
    render_resolution: list[int] = Field(default_factory=lambda: [800, 600])
    render_background: str = "white"
    execution_timeout: int = 30


class StorageConfig(BaseModel):
    """Storage configuration."""
    db_path: str = "db/process.sqlite"
    artifacts_dir: str = "artifacts"
    cad_dir: str = "artifacts/cad"
    images_dir: str = "artifacts/images"
    renders_dir: str = "artifacts/renders"
    wandb_enabled: bool = False
    wandb_project: str = "agentic-cad"
    wandb_entity: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "rich"
    log_file: Optional[str] = None


class Config(BaseModel):
    """Root configuration object."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cad: CADConfig = Field(default_factory=CADConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# =============================================================================
# Configuration Loading
# =============================================================================

def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in config values.
    
    Supports format: ${VAR_NAME:default_value}
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        
        return re.sub(pattern, replace, value)
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    
    return value


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
    
    Returns:
        Config object with all settings.
    """
    # Default config path
    if config_path is None:
        # Look for config in standard locations
        candidates = [
            Path("configs/default.yaml"),
            Path(__file__).parent.parent / "configs" / "default.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
    
    if config_path is None or not Path(config_path).exists():
        # Return default config if no file found
        return Config()
    
    # Load YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return Config()
    
    # Substitute environment variables
    config_dict = _substitute_env_vars(raw_config)
    
    # Parse into Config model
    return Config(**config_dict)


# =============================================================================
# Global Config Instance
# =============================================================================

_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config

