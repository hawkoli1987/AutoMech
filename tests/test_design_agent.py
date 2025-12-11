"""
Tests for Step 1.4: LangGraph Design Agent

Verifies:
- State conversion between Pydantic and TypedDict
- Node function signatures and return values
- Graph construction and compilation
- Agent execution flow

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
            text_desc="A flange with 100mm diameter",
            gt_param_spec={"base_diameter": 100},
            pred_param_spec={"base_diameter": 95},
            param_score=0.9,
            param_feedback="Close match",
            iteration=1,
            done=False,
            metadata=GraphStateMetadata(
                sample_id="flange_001",
                category=CADCategory.FLANGE,
                stl_path="/path/to/stl",
                run_id="run_001",
            ),
        )
        
        agent_state = graph_state_to_agent_state(gs)
        
        assert agent_state["text_desc"] == "A flange with 100mm diameter"
        assert agent_state["gt_param_spec"] == {"base_diameter": 100}
        assert agent_state["pred_param_spec"] == {"base_diameter": 95}
        assert agent_state["param_score"] == 0.9
        assert agent_state["sample_id"] == "flange_001"
        assert agent_state["category"] == "Flange"
        assert agent_state["iteration"] == 1
        assert agent_state["done"] is False
    
    def test_agent_state_to_graph_state(self):
        """Test converting AgentState back to GraphState."""
        from src.agents.design_agent import AgentState, agent_state_to_graph_state
        from src.schemas import CADCategory
        
        agent_state = AgentState(
            text_desc="A gear with 20 teeth",
            gt_param_spec={"teeth_number": 20},
            pred_param_spec={"teeth_number": 20},
            param_score=1.0,
            param_feedback="Perfect match",
            cad_file="/path/to/cad.step",
            render_image="/path/to/render.png",
            vlm_score=0.95,
            vlm_feedback="Good visual quality",
            iteration=2,
            done=True,
            sample_id="gear_001",
            category="Gear",
            stl_path="/path/to/stl",
            run_id="run_002",
            iteration_history=[],
        )
        
        gs = agent_state_to_graph_state(agent_state)
        
        assert gs.text_desc == "A gear with 20 teeth"
        assert gs.gt_param_spec == {"teeth_number": 20}
        assert gs.param_score == 1.0
        assert gs.vlm_score == 0.95
        assert gs.metadata.sample_id == "gear_001"
        assert gs.metadata.category == CADCategory.GEAR
        assert gs.done is True
    
    def test_roundtrip_conversion(self):
        """Test that state survives roundtrip conversion."""
        from src.schemas import GraphState, GraphStateMetadata, CADCategory
        from src.agents.design_agent import (
            graph_state_to_agent_state,
            agent_state_to_graph_state,
        )
        
        original = GraphState(
            text_desc="A nut with 24mm inner diameter",
            gt_param_spec={"inner_diameter": 24},
            metadata=GraphStateMetadata(
                sample_id="nut_001",
                category=CADCategory.NUT,
            ),
        )
        
        agent_state = graph_state_to_agent_state(original)
        roundtrip = agent_state_to_graph_state(agent_state)
        
        assert roundtrip.text_desc == original.text_desc
        assert roundtrip.gt_param_spec == original.gt_param_spec
        assert roundtrip.metadata.sample_id == original.metadata.sample_id
        assert roundtrip.metadata.category == original.metadata.category


class TestCategorySchemas:
    """Test category parameter schemas."""
    
    def test_all_categories_have_schemas(self):
        """Test that all categories have defined schemas."""
        from src.agents.design_agent import CATEGORY_PARAM_SCHEMAS
        from src.schemas import CADCategory
        
        for category in CADCategory:
            assert category.value in CATEGORY_PARAM_SCHEMAS, f"Missing schema for {category.value}"
    
    def test_flange_schema_has_required_fields(self):
        """Test Flange schema has expected fields."""
        from src.agents.design_agent import CATEGORY_PARAM_SCHEMAS
        
        schema = CATEGORY_PARAM_SCHEMAS["Flange"]
        properties = schema.get("properties", {})
        
        expected = ["base_diameter", "base_height", "outer_diameter", "inner_diameter", "flange_height"]
        for field in expected:
            assert field in properties, f"Missing field: {field}"
    
    def test_shaft_schema_is_array(self):
        """Test Shaft schema is array type."""
        from src.agents.design_agent import CATEGORY_PARAM_SCHEMAS
        
        schema = CATEGORY_PARAM_SCHEMAS["Shaft"]
        assert schema["type"] == "array"


class TestHeuristicScoring:
    """Test heuristic fallback scoring."""
    
    def test_dict_exact_match(self):
        """Test heuristic scoring with exact match."""
        from src.agents.design_agent import _heuristic_param_score
        
        gt = {"a": 100, "b": 50}
        pred = {"a": 100, "b": 50}
        
        score = _heuristic_param_score(gt, pred)
        assert score == 1.0
    
    def test_dict_partial_match(self):
        """Test heuristic scoring with partial match."""
        from src.agents.design_agent import _heuristic_param_score
        
        gt = {"a": 100, "b": 50}
        pred = {"a": 95, "b": 50}  # 5% off on 'a'
        
        score = _heuristic_param_score(gt, pred)
        assert 0.9 < score < 1.0
    
    def test_dict_missing_key(self):
        """Test heuristic scoring with missing key."""
        from src.agents.design_agent import _heuristic_param_score
        
        gt = {"a": 100, "b": 50}
        pred = {"a": 100}  # Missing 'b'
        
        score = _heuristic_param_score(gt, pred)
        assert score == 0.5  # One match, one missing
    
    def test_array_exact_match(self):
        """Test heuristic scoring for array (Shaft) with exact match."""
        from src.agents.design_agent import _heuristic_param_score
        
        gt = [[28.8, 10.8], [26.3, 17.8]]
        pred = [[28.8, 10.8], [26.3, 17.8]]
        
        score = _heuristic_param_score(gt, pred)
        assert score == 1.0
    
    def test_array_different_lengths(self):
        """Test heuristic scoring for arrays with different lengths."""
        from src.agents.design_agent import _heuristic_param_score
        
        gt = [[28.8, 10.8], [26.3, 17.8]]
        pred = [[28.8, 10.8]]  # Missing second section
        
        score = _heuristic_param_score(gt, pred)
        assert score == 0.3  # Penalty for length mismatch
    
    def test_empty_specs(self):
        """Test heuristic scoring with empty specs."""
        from src.agents.design_agent import _heuristic_param_score
        
        score = _heuristic_param_score({}, {})
        assert score == 0.0


class TestNodeFunctions:
    """Test individual node functions (mocked)."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        with patch("src.agents.design_agent.get_llm_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    def test_generate_param_spec_returns_dict(self, mock_llm_client):
        """Test generate_param_spec returns correct structure."""
        from src.agents.design_agent import generate_param_spec, AgentState
        
        mock_llm_client.generate_json.return_value = {"base_diameter": 100}
        
        state = AgentState(
            text_desc="A flange with 100mm diameter",
            gt_param_spec={"base_diameter": 100},
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="flange_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = generate_param_spec(state)
        
        assert "pred_param_spec" in result
        assert result["pred_param_spec"] == {"base_diameter": 100}
    
    def test_judge_param_spec_returns_score(self, mock_llm_client):
        """Test judge_param_spec returns score and feedback."""
        from src.agents.design_agent import judge_param_spec, AgentState
        
        mock_llm_client.generate_json.return_value = {
            "overall_score": 0.85,
            "parameter_scores": {"base_diameter": 0.9},
            "feedback": "Close match",
        }
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={"base_diameter": 100},
            pred_param_spec={"base_diameter": 95},
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="flange_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = judge_param_spec(state)
        
        assert "param_score" in result
        assert "param_feedback" in result
        assert result["param_score"] == 0.85
    
    def test_judge_param_spec_no_prediction(self, mock_llm_client):
        """Test judge_param_spec handles missing prediction."""
        from src.agents.design_agent import judge_param_spec, AgentState
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={"base_diameter": 100},
            pred_param_spec=None,  # No prediction
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            sample_id="flange_001",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = judge_param_spec(state)
        
        assert result["param_score"] == 0.0
        assert "No parameters" in result["param_feedback"]


class TestDecideNextStep:
    """Test decision logic for iteration control."""
    
    def test_continue_when_scores_low(self):
        """Test that agent continues when scores are below threshold."""
        from src.agents.design_agent import decide_next_step, AgentState
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={},
            pred_param_spec={},
            param_score=0.5,  # Below threshold
            param_feedback="",
            cad_file=None,
            render_image=None,
            vlm_score=0.5,
            vlm_feedback="",
            iteration=1,
            done=False,
            sample_id="test",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = decide_next_step(state)
        
        assert result["done"] is False
        assert result["iteration"] == 2
    
    def test_stop_when_scores_high(self):
        """Test that agent stops when scores meet threshold."""
        from src.agents.design_agent import decide_next_step, AgentState
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={},
            pred_param_spec={},
            param_score=0.95,  # Above threshold
            param_feedback="",
            cad_file=None,
            render_image=None,
            vlm_score=0.9,  # Above threshold
            vlm_feedback="",
            iteration=1,
            done=False,
            sample_id="test",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = decide_next_step(state)
        
        assert result["done"] is True
    
    def test_stop_at_max_iterations(self):
        """Test that agent stops at max iterations."""
        from src.agents.design_agent import decide_next_step, AgentState
        from src.config import get_config
        
        config = get_config()
        max_iter = config.agent.max_iterations
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={},
            pred_param_spec={},
            param_score=0.5,  # Below threshold
            param_feedback="",
            cad_file=None,
            render_image=None,
            vlm_score=0.5,  # Below threshold
            vlm_feedback="",
            iteration=max_iter,  # At max
            done=False,
            sample_id="test",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = decide_next_step(state)
        
        assert result["done"] is True
    
    def test_history_updated(self):
        """Test that iteration history is updated."""
        from src.agents.design_agent import decide_next_step, AgentState
        
        state = AgentState(
            text_desc="A flange",
            gt_param_spec={},
            pred_param_spec={"base_diameter": 100},
            param_score=0.8,
            param_feedback="Good",
            cad_file=None,
            render_image=None,
            vlm_score=0.7,
            vlm_feedback="OK",
            iteration=1,
            done=False,
            sample_id="test",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = decide_next_step(state)
        
        assert len(result["iteration_history"]) == 1
        entry = result["iteration_history"][0]
        assert entry["iteration"] == 1
        assert entry["param_score"] == 0.8
        assert entry["vlm_score"] == 0.7


class TestGraphConstruction:
    """Test LangGraph construction."""
    
    def test_build_design_agent(self):
        """Test that graph compiles without error."""
        from src.agents.design_agent import build_design_agent
        
        graph = build_design_agent()
        assert graph is not None
    
    def test_design_agent_class(self):
        """Test DesignAgent class initialization."""
        from src.agents.design_agent import DesignAgent
        
        agent = DesignAgent()
        assert agent.graph is not None


class TestConditionalEdges:
    """Test conditional edge logic."""
    
    def test_should_continue_when_not_done(self):
        """Test should_continue returns generate_param_spec when not done."""
        from src.agents.design_agent import should_continue, AgentState
        
        state = AgentState(
            text_desc="",
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
            sample_id="",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = should_continue(state)
        assert result == "generate_param_spec"
    
    def test_should_continue_when_done(self):
        """Test should_continue returns end when done."""
        from src.agents.design_agent import should_continue, AgentState
        
        state = AgentState(
            text_desc="",
            gt_param_spec={},
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=True,
            sample_id="",
            category="Flange",
            stl_path=None,
            run_id=None,
            iteration_history=[],
        )
        
        result = should_continue(state)
        assert result == "end"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

