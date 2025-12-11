"""
Tests for Step 1.6: Storage Layer

Verifies:
- ORM model creation
- Database operations (CRUD)
- Relationships between entities
- Statistics queries
"""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestDatabaseInit:
    """Test database initialization."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        yield str(db_path)
        shutil.rmtree(tmp_dir)
    
    def test_database_creation(self, temp_db):
        """Test database is created."""
        from src.storage.database import DatabaseManager
        
        db = DatabaseManager(db_path=temp_db)
        
        assert Path(temp_db).exists()
    
    def test_tables_created(self, temp_db):
        """Test all tables are created."""
        from src.storage.database import DatabaseManager
        from sqlalchemy import inspect
        
        db = DatabaseManager(db_path=temp_db)
        
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        assert "samples" in tables
        assert "runs" in tables
        assert "nodes" in tables
        assert "artifacts" in tables


class TestSampleOperations:
    """Test sample CRUD operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database manager."""
        from src.storage.database import DatabaseManager
        
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        db_manager = DatabaseManager(db_path=str(db_path))
        yield db_manager
        shutil.rmtree(tmp_dir)
    
    def test_upsert_sample_create(self, db):
        """Test creating a new sample."""
        sample = db.upsert_sample(
            sample_id="flange_00001",
            category="Flange",
            text_desc="A flange with 100mm diameter",
            gt_param_spec={"base_diameter": 100},
            stl_path="/path/to/flange.stl",
        )
        
        assert sample.id == "flange_00001"
        assert sample.category == "Flange"
        assert sample.text_desc == "A flange with 100mm diameter"
    
    def test_upsert_sample_update(self, db):
        """Test updating an existing sample."""
        # Create
        db.upsert_sample(
            sample_id="flange_00001",
            category="Flange",
            text_desc="Original description",
        )
        
        # Update
        sample = db.upsert_sample(
            sample_id="flange_00001",
            category="Flange",
            text_desc="Updated description",
        )
        
        assert sample.text_desc == "Updated description"
    
    def test_get_sample(self, db):
        """Test retrieving a sample."""
        db.upsert_sample(
            sample_id="gear_00001",
            category="Gear",
            gt_param_spec={"teeth_number": 20},
        )
        
        sample = db.get_sample("gear_00001")
        
        assert sample is not None
        assert sample.category == "Gear"
        assert sample.gt_param_spec["teeth_number"] == 20
    
    def test_get_sample_not_found(self, db):
        """Test retrieving non-existent sample."""
        sample = db.get_sample("nonexistent")
        assert sample is None


class TestRunOperations:
    """Test run CRUD operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database manager with a sample."""
        from src.storage.database import DatabaseManager
        
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        db_manager = DatabaseManager(db_path=str(db_path))
        
        # Create a sample for runs
        db_manager.upsert_sample(
            sample_id="flange_00001",
            category="Flange",
        )
        
        yield db_manager
        shutil.rmtree(tmp_dir)
    
    def test_create_run(self, db):
        """Test creating a new run."""
        run = db.create_run(
            sample_id="flange_00001",
            run_id="test_run_001",
        )
        
        assert run.id == "test_run_001"
        assert run.sample_id == "flange_00001"
        assert run.started_at is not None
    
    def test_create_run_auto_id(self, db):
        """Test creating run with auto-generated ID."""
        run = db.create_run(sample_id="flange_00001")
        
        assert run.id is not None
        assert len(run.id) == 8  # UUID prefix
    
    def test_complete_run(self, db):
        """Test completing a run."""
        run = db.create_run(
            sample_id="flange_00001",
            run_id="test_run_002",
        )
        
        db.complete_run(
            run_id="test_run_002",
            success=True,
            final_param_score=0.95,
            final_vlm_score=0.9,
            total_iterations=3,
            final_pred_param_spec={"base_diameter": 100},
        )
        
        # Retrieve and verify
        completed_run = db.get_run("test_run_002")
        assert completed_run.success is True
        assert completed_run.final_param_score == 0.95
        assert completed_run.total_iterations == 3
        assert completed_run.duration_seconds is not None
    
    def test_get_runs_for_sample(self, db):
        """Test getting all runs for a sample."""
        db.create_run(sample_id="flange_00001", run_id="run1")
        db.create_run(sample_id="flange_00001", run_id="run2")
        db.create_run(sample_id="flange_00001", run_id="run3")
        
        runs = db.get_runs_for_sample("flange_00001")
        
        assert len(runs) == 3


class TestNodeExecutionOperations:
    """Test node execution logging."""
    
    @pytest.fixture
    def db_with_run(self):
        """Create a database with a sample and run."""
        from src.storage.database import DatabaseManager
        
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        db_manager = DatabaseManager(db_path=str(db_path))
        
        db_manager.upsert_sample(sample_id="test_sample", category="Flange")
        db_manager.create_run(sample_id="test_sample", run_id="test_run")
        
        yield db_manager
        shutil.rmtree(tmp_dir)
    
    def test_log_node_execution(self, db_with_run):
        """Test logging a node execution."""
        node = db_with_run.log_node_execution(
            run_id="test_run",
            node_name="generate_param_spec",
            iteration=0,
            output_delta={"pred_param_spec": {"base_diameter": 100}},
            duration_ms=150.5,
            success=True,
        )
        
        assert node.node_name == "generate_param_spec"
        assert node.iteration == 0
        assert node.success is True
    
    def test_log_failed_node(self, db_with_run):
        """Test logging a failed node execution."""
        node = db_with_run.log_node_execution(
            run_id="test_run",
            node_name="generate_cad",
            iteration=1,
            success=False,
            error_message="CadQuery not available",
        )
        
        assert node.success is False
        assert "CadQuery" in node.error_message
    
    def test_get_nodes_for_run(self, db_with_run):
        """Test getting all nodes for a run."""
        db_with_run.log_node_execution(run_id="test_run", node_name="node1")
        db_with_run.log_node_execution(run_id="test_run", node_name="node2")
        db_with_run.log_node_execution(run_id="test_run", node_name="node3")
        
        nodes = db_with_run.get_nodes_for_run("test_run")
        
        assert len(nodes) == 3
        assert nodes[0].node_name == "node1"


class TestArtifactOperations:
    """Test artifact logging."""
    
    @pytest.fixture
    def db_with_run(self):
        """Create a database with a sample and run."""
        from src.storage.database import DatabaseManager
        
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        db_manager = DatabaseManager(db_path=str(db_path))
        
        db_manager.upsert_sample(sample_id="test_sample", category="Flange")
        db_manager.create_run(sample_id="test_sample", run_id="test_run")
        
        yield db_manager, tmp_dir
        shutil.rmtree(tmp_dir)
    
    def test_log_artifact(self, db_with_run):
        """Test logging an artifact."""
        db, tmp_dir = db_with_run
        
        artifact = db.log_artifact(
            run_id="test_run",
            artifact_type="cad",
            file_path="/path/to/model.step",
            iteration=0,
        )
        
        assert artifact.artifact_type == "cad"
        assert artifact.file_path == "/path/to/model.step"
    
    def test_log_artifact_with_file_size(self, db_with_run):
        """Test logging artifact with actual file size."""
        db, tmp_dir = db_with_run
        
        # Create a test file
        test_file = Path(tmp_dir) / "test.txt"
        test_file.write_text("Hello, World!")
        
        artifact = db.log_artifact(
            run_id="test_run",
            artifact_type="render",
            file_path=str(test_file),
            iteration=0,
        )
        
        assert artifact.file_size_bytes == 13  # "Hello, World!"
    
    def test_get_artifacts_for_run(self, db_with_run):
        """Test getting all artifacts for a run."""
        db, _ = db_with_run
        
        db.log_artifact(run_id="test_run", artifact_type="cad", file_path="/a.step")
        db.log_artifact(run_id="test_run", artifact_type="render", file_path="/a.png")
        
        artifacts = db.get_artifacts_for_run("test_run")
        
        assert len(artifacts) == 2


class TestStatistics:
    """Test statistics queries."""
    
    @pytest.fixture
    def populated_db(self):
        """Create a populated database."""
        from src.storage.database import DatabaseManager
        
        tmp_dir = tempfile.mkdtemp()
        db_path = Path(tmp_dir) / "test.sqlite"
        db_manager = DatabaseManager(db_path=str(db_path))
        
        # Add samples
        db_manager.upsert_sample(sample_id="sample1", category="Flange")
        db_manager.upsert_sample(sample_id="sample2", category="Gear")
        
        # Add runs
        db_manager.create_run(sample_id="sample1", run_id="run1")
        db_manager.complete_run(run_id="run1", success=True)
        
        db_manager.create_run(sample_id="sample1", run_id="run2")
        db_manager.complete_run(run_id="run2", success=False)
        
        # Add nodes
        db_manager.log_node_execution(run_id="run1", node_name="node1")
        db_manager.log_node_execution(run_id="run1", node_name="node2")
        
        # Add artifacts
        db_manager.log_artifact(run_id="run1", artifact_type="cad", file_path="/a.step")
        
        yield db_manager
        shutil.rmtree(tmp_dir)
    
    def test_get_stats(self, populated_db):
        """Test getting database statistics."""
        stats = populated_db.get_stats()
        
        assert stats["total_samples"] == 2
        assert stats["total_runs"] == 2
        assert stats["successful_runs"] == 1
        assert stats["total_nodes"] == 2
        assert stats["total_artifacts"] == 1


class TestSingletonPattern:
    """Test singleton database pattern."""
    
    def test_get_database_returns_same_instance(self):
        """Test that get_database returns singleton."""
        from src.storage.database import get_database, init_database
        import tempfile
        import shutil
        
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(tmp_dir) / "singleton_test.sqlite"
            init_database(str(db_path))
            
            db1 = get_database()
            db2 = get_database()
            
            assert db1 is db2
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

