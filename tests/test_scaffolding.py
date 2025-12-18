"""
Tests for Step 1.1: Project Scaffolding

Verifies:
- Package structure imports correctly
- Pydantic schemas validate properly
- Configuration loads with env var substitution
"""

import os
import pytest
from pathlib import Path


class TestPackageStructure:
    """Test that all packages can be imported."""
    
    def test_import_src(self):
        """Test importing the main src package."""
        import src
        assert hasattr(src, '__version__')
        assert src.__version__ == "0.1.0"
    
    def test_import_subpackages(self):
        """Test importing all subpackages."""
        import src.agents
        import src.cad
        import src.data_loop
        import src.judges
        import src.storage
        import src.utils
    
    def test_import_schemas(self):
        """Test importing schemas module."""
        from src.schemas import (
            GraphState,
            GraphStateMetadata,
            CadQueryCodeDesign,
            ParamJudgeResult,
            VLMJudgeResult,
            CADResult,
            LLM4CADSample,
        )
    
    def test_import_config(self):
        """Test importing config module."""
        from src.config import (
            Config,
            LLMConfig,
            VLMConfig,
            AgentConfig,
            DataConfig,
            CADConfig,
            StorageConfig,
            LoggingConfig,
            load_config,
            get_config,
        )


class TestSchemas:
    """Test Pydantic schema validation."""
    
    def test_cadquery_code_design_valid(self):
        """Test valid CadQueryCodeDesign creation."""
        from src.schemas import CadQueryCodeDesign
        
        design = CadQueryCodeDesign(
            description="A simple box",
            code="result = cq.Workplane('XY').box(10, 10, 10)",
            entry_point="result",
            required_imports=["cadquery as cq"],
        )
        assert design.description == "A simple box"
        assert "box(10, 10, 10)" in design.code
        assert design.entry_point == "result"
    
    def test_cadquery_code_design_minimal(self):
        """Test minimal CadQueryCodeDesign with defaults."""
        from src.schemas import CadQueryCodeDesign
        
        design = CadQueryCodeDesign(
            description="A cylinder",
            code="result = cq.Workplane('XY').circle(5).extrude(10)",
            nut_height=19.0,
            inner_diameter=24.0
        )
        assert design.entry_point == "result"
        assert len(design.required_imports) > 0
    
    def test_graph_state_metadata(self):
        """Test GraphStateMetadata creation."""
        from src.schemas import GraphStateMetadata
        
        meta = GraphStateMetadata(
            sample_id="bracket_00001",
            stl_path="/path/to/bracket.stl",
            run_id="run_001"
        )
        assert meta.sample_id == "bracket_00001"
    
    def test_graph_state_creation(self):
        """Test GraphState creation with all fields."""
        from src.schemas import GraphState, GraphStateMetadata
        
        state = GraphState(
            text_desc="A mounting bracket with two holes",
            gt_param_spec=None,
            metadata=GraphStateMetadata(
                sample_id="bracket_00001",
            )
        )
        assert state.text_desc == "A mounting bracket with two holes"
        assert state.iteration == 0
        assert state.done is False
        assert state.pred_param_spec is None
    
    def test_llm4cad_sample_to_graph_state(self):
        """Test converting LLM4CADSample to GraphState."""
        from src.schemas import LLM4CADSample
        
        sample = LLM4CADSample(
            sample_id="bracket_00001",
            text_desc="A mounting bracket with two holes",
            gt_param_spec=None,
            stl_path="/path/to/bracket_00001.stl"
        )
        
        state = sample.to_graph_state(run_id="test_run")
        
        assert state.text_desc == sample.text_desc
        assert state.gt_param_spec == sample.gt_param_spec
        assert state.metadata.sample_id == "bracket_00001"
        assert state.metadata.run_id == "test_run"
        assert state.iteration == 0
        assert state.done is False
    
    def test_param_judge_result_score_bounds(self):
        """Test ParamJudgeResult score validation."""
        from src.schemas import ParamJudgeResult
        
        # Valid score
        result = ParamJudgeResult(
            overall_score=0.85,
            parameter_scores={"diameter": 0.9, "height": 0.8},
            feedback="Good match"
        )
        assert result.overall_score == 0.85
        
        # Invalid score should raise
        with pytest.raises(Exception):
            ParamJudgeResult(overall_score=1.5, feedback="")
    
    def test_vlm_judge_result(self):
        """Test VLMJudgeResult creation."""
        from src.schemas import VLMJudgeResult
        
        result = VLMJudgeResult(
            overall_score=0.9,
            geometric_score=0.95,
            completeness_score=0.85,
            quality_score=0.90,
            feedback="Excellent visual quality"
        )
        assert result.overall_score == 0.9
    
    def test_cad_result_success(self):
        """Test CADResult for successful generation."""
        from src.schemas import CADResult
        
        result = CADResult(
            success=True,
            cad_file="/path/to/output.step",
            render_image="/path/to/render.png",
            volume=12345.67
        )
        assert result.success is True
        assert result.error_message is None
    
    def test_cad_result_failure(self):
        """Test CADResult for failed generation."""
        from src.schemas import CADResult
        
        result = CADResult(
            success=False,
            error_message="Invalid geometry: non-manifold edges"
        )
        assert result.success is False
        assert result.cad_file is None


class TestConfig:
    """Test configuration loading."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        from src.config import Config
        
        config = Config()
        assert config.llm.api_base == "http://localhost:8001"
        assert config.agent.max_iterations == 5
        assert config.agent.param_score_threshold == 0.85
    
    def test_load_config_from_file(self):
        """Test loading config from YAML file."""
        from src.config import load_config
        
        config = load_config("configs/default.yaml")
        assert config.llm is not None
        assert config.agent is not None
        assert config.storage.db_path == "db/process.sqlite"
    
    def test_env_var_substitution(self):
        """Test environment variable substitution in config."""
        from src.config import _substitute_env_vars
        
        # Set test env var
        os.environ["TEST_API_BASE"] = "http://test:9000"
        
        # Test substitution
        result = _substitute_env_vars("${TEST_API_BASE}")
        assert result == "http://test:9000"
        
        # Test with default
        result = _substitute_env_vars("${NONEXISTENT_VAR:default_value}")
        assert result == "default_value"
        
        # Cleanup
        del os.environ["TEST_API_BASE"]
    
    def test_env_var_in_dict(self):
        """Test env var substitution in nested dict."""
        from src.config import _substitute_env_vars
        
        os.environ["TEST_VAR"] = "test_value"
        
        config_dict = {
            "llm": {
                "api_base": "${TEST_VAR}",
                "nested": {
                    "value": "${TEST_VAR:default}"
                }
            }
        }
        
        result = _substitute_env_vars(config_dict)
        assert result["llm"]["api_base"] == "test_value"
        assert result["llm"]["nested"]["value"] == "test_value"
        
        del os.environ["TEST_VAR"]
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton."""
        from src.config import get_config, set_config, Config
        
        # Reset global config
        set_config(Config())
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2


class TestDirectoryStructure:
    """Test that required directories exist."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    def test_src_directory_exists(self, project_root):
        """Test src/ directory exists."""
        assert (project_root / "src").is_dir()
    
    def test_subpackages_exist(self, project_root):
        """Test all subpackages have __init__.py."""
        packages = ["agents", "cad", "data_loop", "judges", "storage", "utils"]
        for pkg in packages:
            init_file = project_root / "src" / pkg / "__init__.py"
            assert init_file.exists(), f"Missing: {init_file}"
    
    def test_configs_directory_exists(self, project_root):
        """Test configs/ directory exists."""
        assert (project_root / "configs").is_dir()
        assert (project_root / "configs" / "default.yaml").exists()
    
    def test_artifacts_directories_exist(self, project_root):
        """Test artifacts directories exist."""
        assert (project_root / "artifacts").is_dir()
        assert (project_root / "artifacts" / "cad").is_dir()
        assert (project_root / "artifacts" / "images").is_dir()
    
    def test_db_directory_exists(self, project_root):
        """Test db/ directory exists."""
        assert (project_root / "db").is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

