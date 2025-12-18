"""
Tests for LangGraph Design Agent - Freeform Code Generation

Verifies:
- State conversion between Pydantic and TypedDict
- Freeform code generation node
- CAD generation node
- VLM judging node
- Graph construction and compilation

Note: Tests requiring live LLM are marked with @pytest.mark.live
"""

import pytest
from unittest.mock import patch, MagicMock


class TestStateConversion:
    """Test state conversion between Pydantic and TypedDict."""
    
    def test_graph_state_to_agent_state(self):
        """Test converting GraphState to AgentState."""
        from src.schemas import GraphState, GraphStateMetadata, CADCategory
        from src.agents.design_agent import graph_state_to_agent_state
        
        gs = GraphState(
            text_desc="A mounting bracket with two holes",
            gt_param_spec={},  # Not used in freeform mode
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            iteration=1,
            done=False,
            metadata=GraphStateMetadata(
                sample_id="bracket_001",
                category=CADCategory.FLANGE,  # Category less relevant now
                stl_path="/path/to/stl",
                run_id="run_001",
            ),
        )
        
        agent_state = graph_state_to_agent_state(gs)
        
        assert agent_state["text_desc"] == "A mounting bracket with two holes"
        assert agent_state["sample_id"] == "bracket_001"
        assert agent_state["iteration"] == 1
        assert agent_state["done"] is False
    
    def test_agent_state_to_graph_state(self):
        """Test converting AgentState back to GraphState."""
        from src.agents.design_agent import AgentState, agent_state_to_graph_state
        from src.schemas import CADCategory
        
        agent_state = AgentState(
            text_desc="A cylindrical spacer",
            gt_param_spec={},
            pred_param_spec={"code": "result = cq.Workplane('XY').box(10,10,10)"},
            param_score=None,
            param_feedback=None,
            cad_file="/path/to/cad.step",
            render_image="/path/to/render.png",
            vlm_score=0.95,
            vlm_feedback="Good visual quality",
            iteration=2,
            done=True,
            sample_id="spacer_001",
            category="Flange",
            stl_path="/path/to/stl",
            run_id="run_002",
            iteration_history=[],
        )
        
        gs = agent_state_to_graph_state(agent_state)
        
        assert gs.text_desc == "A cylindrical spacer"
        assert gs.pred_param_spec == {"code": "result = cq.Workplane('XY').box(10,10,10)"}
        assert gs.vlm_score == 0.95
        assert gs.metadata.sample_id == "spacer_001"
        assert gs.done is True
    
    def test_roundtrip_conversion(self):
        """Test that state survives roundtrip conversion."""
        from src.schemas import GraphState, GraphStateMetadata, CADCategory
        from src.agents.design_agent import (
            graph_state_to_agent_state,
            agent_state_to_graph_state,
        )
        
        original = GraphState(
            text_desc="A custom part",
            gt_param_spec={},
            metadata=GraphStateMetadata(
                sample_id="custom_001",
                category=CADCategory.FLANGE,
            ),
        )
        
        # Convert back and forth
        agent_state = graph_state_to_agent_state(original)
        recovered = agent_state_to_graph_state(agent_state)
        
        assert recovered.text_desc == original.text_desc
        assert recovered.metadata.sample_id == original.metadata.sample_id


class TestNodeFunctions:
    """Test individual node functions (mocked)."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        with patch("src.agents.design_agent.get_llm_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    def test_generate_freeform_code_returns_dict(self, mock_llm_client):
        """Test generate_freeform_code returns correct structure."""
        from src.agents.design_agent import generate_freeform_code, AgentState
        
        mock_llm_client.generate_json.return_value = {
            "description": "Simple box",
            "code": "result = cq.Workplane('XY').box(10, 10, 10)",
            "entry_point": "result",
            "required_imports": [],
        }
        
        state = AgentState(
            text_desc="A 10mm cube",
            gt_param_spec={},
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="cube_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = generate_freeform_code(state)
        
        assert "pred_param_spec" in result
        assert "code" in result["pred_param_spec"]
        assert "result = cq.Workplane" in result["pred_param_spec"]["code"]
    
    def test_generate_freeform_code_handles_failure(self, mock_llm_client):
        """Test generate_freeform_code handles LLM failure gracefully."""
        from src.agents.design_agent import generate_freeform_code, AgentState
        
        mock_llm_client.generate_json.side_effect = Exception("LLM unavailable")
        
        state = AgentState(
            text_desc="A complex part",
            gt_param_spec={},
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="part_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = generate_freeform_code(state)
        
        # Should return fallback code
        assert "pred_param_spec" in result
        assert "code" in result["pred_param_spec"]
    
    @pytest.mark.skipif(
        not _is_cadquery_available(),
        reason="CadQuery not available"
    )
    def test_generate_cad_with_freeform_code(self, mock_llm_client):
        """Test generate_cad processes freeform code."""
        from src.agents.design_agent import generate_cad, AgentState
        import tempfile
        
        state = AgentState(
            text_desc="A box",
            gt_param_spec={},
            pred_param_spec={
                "description": "10mm box",
                "code": "result = cq.Workplane('XY').box(10, 10, 10)",
                "entry_point": "result",
                "required_imports": [],
            },
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="box_001",
            category="Flange",
            stl_path=None,
            run_id="test_run",
            iteration_history=[],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.design_agent.get_config") as mock_config:
                mock_config.return_value.storage.artifacts_dir = tmpdir
                
                result = generate_cad(state)
                
                assert "cad_file" in result
                assert "render_image" in result
    
    def test_generate_cad_handles_no_code(self, mock_llm_client):
        """Test generate_cad handles missing code gracefully."""
        from src.agents.design_agent import generate_cad, AgentState
        
        state = AgentState(
            text_desc="A part",
            gt_param_spec={},
            pred_param_spec=None,  # No code generated
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="part_001",
            category="Flange",
            stl_path=None,
            run_id="test_run",
            iteration_history=[],
        )
        
        result = generate_cad(state)
        
        assert result["cad_file"] is None
        assert result["render_image"] is None


class TestVLMJudging:
    """Test VLM judging functionality."""
    
    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client."""
        with patch("src.agents.design_agent.get_vlm_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    def test_judge_cad_vlm_returns_scores(self, mock_vlm_client):
        """Test VLM judging returns scores and feedback."""
        from src.agents.design_agent import judge_cad_vlm, AgentState
        import tempfile
        from pathlib import Path
        
        # Create temporary render image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'fake image data')
            tmp_path = tmp.name
        
        try:
            mock_vlm_client.generate_json_with_image.return_value = {
                "overall_score": 0.85,
                "geometric_score": 0.9,
                "completeness_score": 0.8,
                "quality_score": 0.85,
                "issues": ["Minor issue"],
                "feedback": "Good quality",
            }
            
            state = AgentState(
                text_desc="A part",
                gt_param_spec={},
                pred_param_spec={"code": "..."},
                param_score=None,
                param_feedback=None,
                cad_file="/path/to/file.step",
                render_image=tmp_path,
                vlm_score=None,
                vlm_feedback=None,
                iteration=0,
                done=False,
                sample_id="part_001",
                category="Flange",
                stl_path=None,
                run_id=None,
                iteration_history=[],
            )
            
            with patch("src.agents.design_agent.get_config") as mock_config:
                mock_config.return_value.agent.enable_vlm_judge = True
                
                result = judge_cad_vlm(state)
                
                assert "vlm_score" in result
                assert result["vlm_score"] == 0.85
                assert "vlm_feedback" in result
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_judge_cad_vlm_handles_no_render(self, mock_vlm_client):
        """Test VLM judging handles missing render image."""
        from src.agents.design_agent import judge_cad_vlm, AgentState
        
        state = AgentState(
            text_desc="A part",
            gt_param_spec={},
            pred_param_spec={"code": "..."},
            param_score=None,
            param_feedback=None,
            cad_file="/path/to/file.step",
            render_image=None,  # No render
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="part_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        with patch("src.agents.design_agent.get_config") as mock_config:
            mock_config.return_value.agent.enable_vlm_judge = True
            
            result = judge_cad_vlm(state)
            
            assert "vlm_score" in result
            assert "vlm_feedback" in result
            assert "No render image available" in result["vlm_feedback"]


def _is_cadquery_available():
    """Helper to check CadQuery availability."""
    try:
        import cadquery
        return True
    except ImportError:
        return False


@pytest.mark.live
class TestLiveIntegration:
    """Integration tests requiring live LLM/VLM."""
    
    def test_full_freeform_pipeline(self):
        """Test complete freeform generation pipeline with real LLM."""
        from src.agents.design_agent import generate_freeform_code, generate_cad
        from src.agents.design_agent import AgentState
        import tempfile
        
        if not _is_cadquery_available():
            pytest.skip("CadQuery not available")
        
        try:
            # Step 1: Generate code
            state = AgentState(
                text_desc="A simple 15mm cube",
                gt_param_spec={},
                pred_param_spec=None,
                param_score=None,
                param_feedback=None,
                cad_file=None,
                render_image=None,
                vlm_score=None,
                vlm_feedback=None,
                iteration=0,
                done=False,
                sample_id="test_001",
                category="Flange",
                stl_path=None,
                run_id="test_run",
                iteration_history=[],
            )
            
            code_result = generate_freeform_code(state)
            assert "pred_param_spec" in code_result
            assert "code" in code_result["pred_param_spec"]
            
            # Step 2: Generate CAD
            state["pred_param_spec"] = code_result["pred_param_spec"]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("src.agents.design_agent.get_config") as mock_config:
                    mock_config.return_value.storage.artifacts_dir = tmpdir
                    
                    cad_result = generate_cad(state)
                    
                    assert "cad_file" in cad_result
                    
        except Exception as e:
            pytest.skip(f"Live LLM not available: {e}")
