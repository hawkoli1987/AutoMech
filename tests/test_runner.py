"""
Tests for Step 1.7: Dataset Processing Loop

Verifies:
- Runner initialization
- Checkpoint management
- Sample processing flow
- Summary printing
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCheckpointManager:
    """Test checkpoint management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)
    
    def test_checkpoint_creation(self, temp_dir):
        """Test creating new checkpoint."""
        from src.data_loop.runner import CheckpointManager
        
        cp_path = Path(temp_dir) / "checkpoint.json"
        cp = CheckpointManager(str(cp_path))
        
        assert len(cp.processed_samples) == 0
    
    def test_mark_processed(self, temp_dir):
        """Test marking sample as processed."""
        from src.data_loop.runner import CheckpointManager
        
        cp_path = Path(temp_dir) / "checkpoint.json"
        cp = CheckpointManager(str(cp_path))
        
        cp.mark_processed("sample_001")
        cp.mark_processed("sample_002")
        
        assert cp.is_processed("sample_001")
        assert cp.is_processed("sample_002")
        assert not cp.is_processed("sample_003")
    
    def test_checkpoint_persistence(self, temp_dir):
        """Test checkpoint saves and loads correctly."""
        from src.data_loop.runner import CheckpointManager
        
        cp_path = Path(temp_dir) / "checkpoint.json"
        
        # Create and save
        cp1 = CheckpointManager(str(cp_path))
        cp1.mark_processed("sample_001")
        cp1.mark_processed("sample_002")
        
        # Load in new manager
        cp2 = CheckpointManager(str(cp_path))
        
        assert cp2.is_processed("sample_001")
        assert cp2.is_processed("sample_002")
    
    def test_checkpoint_file_format(self, temp_dir):
        """Test checkpoint file format is valid JSON."""
        from src.data_loop.runner import CheckpointManager
        
        cp_path = Path(temp_dir) / "checkpoint.json"
        cp = CheckpointManager(str(cp_path))
        cp.mark_processed("sample_001")
        
        # Read and parse JSON
        with open(cp_path) as f:
            data = json.load(f)
        
        assert "processed_samples" in data
        assert "sample_001" in data["processed_samples"]


class TestDatasetRunnerInit:
    """Test runner initialization."""
    
    def test_runner_import(self):
        """Test runner can be imported."""
        from src.data_loop.runner import DatasetRunner
    
    @pytest.mark.skipif(
        not Path("data/LLM4CAD").exists(),
        reason="LLM4CAD data not available"
    )
    def test_runner_initialization(self):
        """Test runner initializes with data."""
        from src.data_loop.runner import DatasetRunner
        import tempfile
        import shutil
        
        tmp = tempfile.mkdtemp()
        try:
            runner = DatasetRunner(
                data_path="data/LLM4CAD",
                max_samples=5,
                db_path=Path(tmp) / "test.sqlite",
            )
            
            assert runner.agent is not None
            assert runner.loader is not None
            assert runner.db is not None
        finally:
            shutil.rmtree(tmp)


class TestWandBIntegration:
    """Test WandB integration."""
    
    def test_init_wandb_without_wandb(self):
        """Test init_wandb handles missing wandb gracefully."""
        from src.data_loop.runner import init_wandb
        
        with patch.dict('sys.modules', {'wandb': None}):
            # This should not raise
            result = init_wandb("test_project")
            # Result will be None or the run depending on if wandb is installed
    
    def test_log_to_wandb_with_none_run(self):
        """Test logging with None run doesn't crash."""
        from src.data_loop.runner import log_to_wandb
        from src.schemas import GraphState, GraphStateMetadata, CADCategory
        
        state = GraphState(
            text_desc="test",
            gt_param_spec={},
            metadata=GraphStateMetadata(
                sample_id="test",
                category=CADCategory.FLANGE,
            ),
            param_score=0.9,
            vlm_score=0.8,
            iteration=2,
        )
        
        # Should not raise
        log_to_wandb(None, "test_sample", state, 1.0)


class TestRunnerWithMocks:
    """Test runner with mocked dependencies."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        from src.schemas import GraphState, GraphStateMetadata, CADCategory
        
        mock = MagicMock()
        
        # Configure mock to return a final state
        final_state = GraphState(
            text_desc="test",
            gt_param_spec={"base_diameter": 100},
            pred_param_spec={"base_diameter": 95},
            param_score=0.9,
            vlm_score=0.85,
            iteration=2,
            done=True,
            metadata=GraphStateMetadata(
                sample_id="test",
                category=CADCategory.FLANGE,
                run_id="test_run",
            ),
        )
        mock.run.return_value = final_state
        
        return mock
    
    @pytest.fixture
    def mock_sample(self):
        """Create a mock sample."""
        from src.schemas import LLM4CADSample, CADCategory
        
        return LLM4CADSample(
            sample_id="flange_00001",
            category=CADCategory.FLANGE,
            text_desc="A flange with 100mm diameter",
            gt_param_spec={"base_diameter": 100},
            stl_path="/path/to/stl",
        )
    
    def test_process_sample_returns_result(self, temp_dir, mock_agent, mock_sample):
        """Test _process_sample returns correct structure."""
        from src.data_loop.runner import DatasetRunner
        from src.storage.database import init_database
        
        # Initialize database
        db_path = Path(temp_dir) / "test.sqlite"
        init_database(str(db_path))
        
        runner = DatasetRunner(
            data_path=None,  # Will fail if it tries to load, but we patch below
            db_path=str(db_path),
        )
        runner.agent = mock_agent
        
        # Manually set loader to avoid needing data
        runner.loader = MagicMock()
        runner.loader.categories = []
        
        result = runner._process_sample(mock_sample)
        
        assert result["sample_id"] == "flange_00001"
        assert result["category"] == "Flange"
        assert result["param_score"] == 0.9
        assert result["vlm_score"] == 0.85
        assert result["iterations"] == 2
        assert result["success"] is True


class TestConsoleOutput:
    """Test console output functions."""
    
    def test_print_summary_doesnt_crash(self, capsys):
        """Test _print_summary runs without error."""
        from src.data_loop.runner import DatasetRunner
        import tempfile
        import shutil
        
        tmp = tempfile.mkdtemp()
        try:
            # Create runner with minimal setup
            runner = DatasetRunner.__new__(DatasetRunner)
            runner.run_id = "test123"
            runner.enable_wandb = False
            runner.wandb_run = None
            
            # Mock loader
            runner.loader = MagicMock()
            runner.loader.categories = []
            
            # Mock config
            runner.config = MagicMock()
            runner.config.agent.param_score_threshold = 0.85
            runner.config.agent.vlm_score_threshold = 0.8
            runner.config.agent.max_iterations = 5
            
            runner._print_summary(10)
            
            # If we get here, it didn't crash
        finally:
            shutil.rmtree(tmp)
    
    def test_print_final_summary_with_results(self):
        """Test _print_final_summary with sample results."""
        from src.data_loop.runner import DatasetRunner
        
        runner = DatasetRunner.__new__(DatasetRunner)
        runner.results = [
            {"success": True, "param_score": 0.9, "vlm_score": 0.85, "iterations": 2, "duration_seconds": 1.5},
            {"success": False, "param_score": 0.5, "vlm_score": 0.4, "iterations": 5, "duration_seconds": 3.0},
        ]
        
        # Should not crash
        runner._print_final_summary()
    
    def test_print_final_summary_empty_results(self):
        """Test _print_final_summary with no results."""
        from src.data_loop.runner import DatasetRunner
        
        runner = DatasetRunner.__new__(DatasetRunner)
        runner.results = []
        
        # Should not crash
        runner._print_final_summary()


class TestCLI:
    """Test CLI entry point."""
    
    def test_main_function_exists(self):
        """Test main function can be imported."""
        from src.data_loop.runner import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

