"""
Storage Layer - Database Module

SQLite database with SQLAlchemy ORM for storing:
- samples: LLM4CAD sample metadata
- runs: Agent execution runs
- nodes: Per-node execution records
- artifacts: Generated file paths
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def utcnow() -> datetime:
    """Get current UTC time as naive datetime (for SQLite compatibility)."""
    # Using naive datetime for SQLite compatibility
    # SQLite doesn't natively support timezone-aware datetimes
    return datetime.utcnow()  # noqa: DTZ003

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
)

from src.config import get_config


Base = declarative_base()


# =============================================================================
# ORM Models
# =============================================================================

class Sample(Base):
    """
    Represents a sample from the LLM4CAD dataset.
    
    One sample can have multiple runs (different agent executions).
    """
    __tablename__ = "samples"
    
    id = Column(String(64), primary_key=True)  # e.g., "flange_00001"
    category = Column(String(32), nullable=False)  # e.g., "Flange"
    text_desc = Column(Text, nullable=True)
    gt_param_spec = Column(JSON, nullable=True)
    stl_path = Column(String(512), nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    runs = relationship("Run", back_populates="sample", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Sample(id='{self.id}', category='{self.category}')>"


class Run(Base):
    """
    Represents a single agent run on a sample.
    
    Each run can have multiple node executions and produce artifacts.
    """
    __tablename__ = "runs"
    
    id = Column(String(64), primary_key=True)  # UUID
    sample_id = Column(String(64), ForeignKey("samples.id"), nullable=False)
    
    # Final state
    final_param_score = Column(Float, nullable=True)
    final_vlm_score = Column(Float, nullable=True)
    total_iterations = Column(Integer, default=0)
    success = Column(Boolean, default=False)
    
    # Timing
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Final outputs
    final_pred_param_spec = Column(JSON, nullable=True)
    final_cad_file = Column(String(512), nullable=True)
    final_render_image = Column(String(512), nullable=True)
    
    # Metadata
    config_snapshot = Column(JSON, nullable=True)  # Config used for this run
    
    # Relationships
    sample = relationship("Sample", back_populates="runs")
    nodes = relationship("NodeExecution", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Run(id='{self.id}', sample_id='{self.sample_id}', success={self.success})>"


class NodeExecution(Base):
    """
    Represents a single node execution within a run.
    
    Captures inputs, outputs, and timing for each LangGraph node.
    """
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=False)
    
    node_name = Column(String(64), nullable=False)  # e.g., "generate_param_spec"
    iteration = Column(Integer, default=0)
    
    # Inputs/Outputs (stored as JSON)
    input_state = Column(JSON, nullable=True)
    output_delta = Column(JSON, nullable=True)  # What the node changed
    
    # Timing
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    run = relationship("Run", back_populates="nodes")
    
    def __repr__(self):
        return f"<NodeExecution(node='{self.node_name}', iteration={self.iteration})>"


class Artifact(Base):
    """
    Represents a generated artifact (CAD file, render, etc.).
    """
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=False)
    
    artifact_type = Column(String(32), nullable=False)  # "cad", "render", "stl"
    file_path = Column(String(512), nullable=False)
    iteration = Column(Integer, default=0)
    
    # File metadata
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="artifacts")
    
    def __repr__(self):
        return f"<Artifact(type='{self.artifact_type}', path='{self.file_path}')>"


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """
    Manages database connections and provides high-level operations.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database. Uses config if not provided.
        """
        config = get_config()
        
        if db_path is None:
            db_path = config.storage.db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # =========================================================================
    # Sample Operations
    # =========================================================================
    
    def upsert_sample(
        self,
        sample_id: str,
        category: str,
        text_desc: Optional[str] = None,
        gt_param_spec: Optional[dict | list] = None,
        stl_path: Optional[str] = None,
    ) -> Sample:
        """
        Insert or update a sample.
        
        Returns the Sample object.
        """
        with self.get_session() as session:
            sample = session.get(Sample, sample_id)
            
            if sample is None:
                sample = Sample(
                    id=sample_id,
                    category=category,
                    text_desc=text_desc,
                    gt_param_spec=gt_param_spec,
                    stl_path=stl_path,
                )
                session.add(sample)
            else:
                sample.text_desc = text_desc or sample.text_desc
                sample.gt_param_spec = gt_param_spec or sample.gt_param_spec
                sample.stl_path = stl_path or sample.stl_path
            
            session.commit()
            session.refresh(sample)
            
            return sample
    
    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """Get a sample by ID."""
        with self.get_session() as session:
            return session.get(Sample, sample_id)
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    def create_run(
        self,
        sample_id: str,
        run_id: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
    ) -> Run:
        """
        Create a new run for a sample.
        
        Returns the Run object.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        
        with self.get_session() as session:
            run = Run(
                id=run_id,
                sample_id=sample_id,
                config_snapshot=config_snapshot,
                started_at=utcnow(),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            
            return run
    
    def complete_run(
        self,
        run_id: str,
        success: bool,
        final_param_score: Optional[float] = None,
        final_vlm_score: Optional[float] = None,
        total_iterations: int = 0,
        final_pred_param_spec: Optional[dict] = None,
        final_cad_file: Optional[str] = None,
        final_render_image: Optional[str] = None,
    ) -> None:
        """Mark a run as complete with final results."""
        with self.get_session() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")
            
            run.success = success
            run.completed_at = utcnow()
            if run.started_at is not None:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            run.final_param_score = final_param_score
            run.final_vlm_score = final_vlm_score
            run.total_iterations = total_iterations
            run.final_pred_param_spec = final_pred_param_spec
            run.final_cad_file = final_cad_file
            run.final_render_image = final_render_image
            
            session.commit()
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        with self.get_session() as session:
            return session.get(Run, run_id)
    
    def get_runs_for_sample(self, sample_id: str) -> list[Run]:
        """Get all runs for a sample."""
        with self.get_session() as session:
            return session.query(Run).filter(Run.sample_id == sample_id).all()
    
    # =========================================================================
    # Node Execution Operations
    # =========================================================================
    
    def log_node_execution(
        self,
        run_id: str,
        node_name: str,
        iteration: int = 0,
        input_state: Optional[dict] = None,
        output_delta: Optional[dict] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> NodeExecution:
        """Log a node execution."""
        with self.get_session() as session:
            node = NodeExecution(
                run_id=run_id,
                node_name=node_name,
                iteration=iteration,
                input_state=input_state,
                output_delta=output_delta,
                started_at=utcnow(),
                completed_at=utcnow(),
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
            )
            session.add(node)
            session.commit()
            session.refresh(node)
            
            return node
    
    def get_nodes_for_run(self, run_id: str) -> list[NodeExecution]:
        """Get all node executions for a run."""
        with self.get_session() as session:
            return session.query(NodeExecution).filter(
                NodeExecution.run_id == run_id
            ).order_by(NodeExecution.id).all()
    
    # =========================================================================
    # Artifact Operations
    # =========================================================================
    
    def log_artifact(
        self,
        run_id: str,
        artifact_type: str,
        file_path: str,
        iteration: int = 0,
        file_size_bytes: Optional[int] = None,
    ) -> Artifact:
        """Log a generated artifact."""
        # Get file size if not provided
        if file_size_bytes is None:
            path = Path(file_path)
            if path.exists():
                file_size_bytes = path.stat().st_size
        
        with self.get_session() as session:
            artifact = Artifact(
                run_id=run_id,
                artifact_type=artifact_type,
                file_path=file_path,
                iteration=iteration,
                file_size_bytes=file_size_bytes,
            )
            session.add(artifact)
            session.commit()
            session.refresh(artifact)
            
            return artifact
    
    def get_artifacts_for_run(self, run_id: str) -> list[Artifact]:
        """Get all artifacts for a run."""
        with self.get_session() as session:
            return session.query(Artifact).filter(
                Artifact.run_id == run_id
            ).all()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get overall database statistics."""
        with self.get_session() as session:
            return {
                "total_samples": session.query(Sample).count(),
                "total_runs": session.query(Run).count(),
                "successful_runs": session.query(Run).filter(Run.success == True).count(),
                "total_nodes": session.query(NodeExecution).count(),
                "total_artifacts": session.query(Artifact).count(),
            }


# =============================================================================
# Singleton Instance
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database(db_path: Optional[str] = None) -> DatabaseManager:
    """Initialize the database with a specific path."""
    global _db_manager
    _db_manager = DatabaseManager(db_path)
    return _db_manager

