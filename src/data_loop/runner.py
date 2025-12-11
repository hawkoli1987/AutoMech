"""
Dataset Processing Loop

Main orchestration for batch processing LLM4CAD samples through the design agent.
Handles progress tracking, WandB logging, and checkpoint/resume.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.config import get_config, Config
from src.data_loop.loader import LLM4CADLoader, LLM4CADSample
from src.agents.design_agent import DesignAgent, get_design_agent
from src.storage.database import DatabaseManager, get_database, init_database
from src.schemas import GraphState, CADCategory


console = Console()


# =============================================================================
# WandB Integration
# =============================================================================

def init_wandb(
    project: str,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
) -> Optional[object]:
    """
    Initialize Weights & Biases logging.
    
    Returns the wandb run object or None if wandb is disabled/unavailable.
    """
    try:
        import wandb
        
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            reinit=True,
        )
        console.print("[green]WandB initialized[/green]")
        return run
    except ImportError:
        console.print("[yellow]WandB not available, skipping[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]WandB init failed: {e}[/yellow]")
        return None


def log_to_wandb(
    wandb_run,
    sample_id: str,
    final_state: GraphState,
    duration_seconds: float,
) -> None:
    """Log results to WandB."""
    if wandb_run is None:
        return
    
    try:
        import wandb
        
        # Log metrics
        wandb.log({
            "param_score": final_state.param_score or 0,
            "vlm_score": final_state.vlm_score or 0,
            "iterations": final_state.iteration,
            "success": final_state.done and (final_state.param_score or 0) >= 0.85,
            "duration_seconds": duration_seconds,
            "category": final_state.metadata.category.value,
        })
    except Exception as e:
        console.print(f"[yellow]WandB log failed: {e}[/yellow]")


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """Manages checkpoint state for resumable processing."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.processed_samples: set[str] = set()
        self._load()
    
    def _load(self) -> None:
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.processed_samples = set(data.get("processed_samples", []))
            except Exception as e:
                console.print(f"[yellow]Failed to load checkpoint: {e}[/yellow]")
    
    def save(self) -> None:
        """Save checkpoint to disk."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump({
                "processed_samples": list(self.processed_samples),
                "last_updated": datetime.now().isoformat(),
            }, f)
    
    def mark_processed(self, sample_id: str) -> None:
        """Mark a sample as processed."""
        self.processed_samples.add(sample_id)
        self.save()
    
    def is_processed(self, sample_id: str) -> bool:
        """Check if a sample has been processed."""
        return sample_id in self.processed_samples


# =============================================================================
# Main Runner
# =============================================================================

class DatasetRunner:
    """
    Main orchestrator for processing LLM4CAD samples.
    
    Features:
    - Batch processing with progress tracking
    - Database persistence
    - WandB integration
    - Checkpoint/resume support
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        categories: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        checkpoint_path: Optional[str] = None,
        enable_wandb: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the runner.
        
        Args:
            data_path: Path to LLM4CAD data
            categories: Categories to process (None = all)
            max_samples: Max samples to process
            shuffle: Shuffle samples
            checkpoint_path: Path for checkpoint file
            enable_wandb: Enable WandB logging
            db_path: Path to SQLite database
        """
        self.config = get_config()
        
        # Data loader
        self.loader = LLM4CADLoader(
            data_path=data_path,
            categories=categories,
            shuffle=shuffle,
        )
        
        self.max_samples = max_samples
        
        # Checkpoint manager
        if checkpoint_path is None:
            checkpoint_path = "artifacts/checkpoint.json"
        self.checkpoint = CheckpointManager(checkpoint_path)
        
        # Database
        if db_path:
            init_database(db_path)
        self.db = get_database()
        
        # WandB
        self.enable_wandb = enable_wandb
        self.wandb_run = None
        
        # Agent
        self.agent = get_design_agent()
        
        # Run tracking
        self.run_id = str(uuid.uuid4())[:8]
        self.results: list[dict] = []
    
    def run(
        self,
        resume: bool = True,
        callback: Optional[Callable[[LLM4CADSample, GraphState], None]] = None,
    ) -> list[dict]:
        """
        Run the processing loop.
        
        Args:
            resume: Skip already processed samples
            callback: Optional callback after each sample
        
        Returns:
            List of result dicts
        """
        # Initialize WandB
        if self.enable_wandb and self.config.storage.wandb_enabled:
            self.wandb_run = init_wandb(
                project=self.config.storage.wandb_project,
                entity=self.config.storage.wandb_entity,
                run_name=f"run_{self.run_id}",
                config={
                    "max_samples": self.max_samples,
                    "categories": [c.value for c in self.loader.categories],
                    "agent_config": {
                        "param_threshold": self.config.agent.param_score_threshold,
                        "vlm_threshold": self.config.agent.vlm_score_threshold,
                        "max_iterations": self.config.agent.max_iterations,
                    }
                },
            )
        
        # Get samples
        samples = list(self.loader)
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        # Filter already processed if resuming
        if resume:
            samples = [s for s in samples if not self.checkpoint.is_processed(s.sample_id)]
        
        if not samples:
            console.print("[yellow]No samples to process[/yellow]")
            return self.results
        
        # Print summary
        self._print_summary(len(samples))
        
        # Process samples with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {len(samples)} samples...",
                total=len(samples),
            )
            
            for sample in samples:
                try:
                    result = self._process_sample(sample)
                    self.results.append(result)
                    
                    # Callback
                    if callback:
                        callback(sample, result["final_state"])
                    
                    # Mark processed
                    self.checkpoint.mark_processed(sample.sample_id)
                    
                except Exception as e:
                    console.print(f"[red]Error processing {sample.sample_id}: {e}[/red]")
                    self.results.append({
                        "sample_id": sample.sample_id,
                        "success": False,
                        "error": str(e),
                    })
                
                progress.advance(task)
        
        # Print final summary
        self._print_final_summary()
        
        # Close WandB
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        
        return self.results
    
    def _process_sample(self, sample: LLM4CADSample) -> dict:
        """Process a single sample."""
        console.print(f"  [dim]Processing {sample.sample_id}...[/dim]")
        
        # Insert sample into database
        self.db.upsert_sample(
            sample_id=sample.sample_id,
            category=sample.category.value,
            text_desc=sample.text_desc,
            gt_param_spec=sample.gt_param_spec,
            stl_path=sample.stl_path,
        )
        
        # Create run record
        config_snapshot = {
            "param_threshold": self.config.agent.param_score_threshold,
            "vlm_threshold": self.config.agent.vlm_score_threshold,
            "max_iterations": self.config.agent.max_iterations,
        }
        run = self.db.create_run(
            sample_id=sample.sample_id,
            config_snapshot=config_snapshot,
        )
        
        # Convert to initial state
        initial_state = sample.to_graph_state(run_id=run.id)
        
        # Run agent
        start_time = time.time()
        final_state = self.agent.run(initial_state)
        duration = time.time() - start_time
        
        # Determine success
        param_score = final_state.param_score or 0
        vlm_score = final_state.vlm_score or 0
        success = (
            param_score >= self.config.agent.param_score_threshold and
            vlm_score >= self.config.agent.vlm_score_threshold
        )
        
        # Complete run record
        self.db.complete_run(
            run_id=run.id,
            success=success,
            final_param_score=param_score,
            final_vlm_score=vlm_score,
            total_iterations=final_state.iteration,
            final_pred_param_spec=final_state.pred_param_spec,
            final_cad_file=final_state.cad_file,
            final_render_image=final_state.render_image,
        )
        
        # Log to WandB
        log_to_wandb(self.wandb_run, sample.sample_id, final_state, duration)
        
        return {
            "sample_id": sample.sample_id,
            "category": sample.category.value,
            "success": success,
            "param_score": param_score,
            "vlm_score": vlm_score,
            "iterations": final_state.iteration,
            "duration_seconds": duration,
            "run_id": run.id,
            "final_state": final_state,
        }
    
    def _print_summary(self, num_samples: int) -> None:
        """Print run configuration summary."""
        table = Table(title="Dataset Processing Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Run ID", self.run_id)
        table.add_row("Samples to process", str(num_samples))
        table.add_row("Categories", ", ".join(c.value for c in self.loader.categories))
        table.add_row("Param threshold", f"{self.config.agent.param_score_threshold:.2f}")
        table.add_row("VLM threshold", f"{self.config.agent.vlm_score_threshold:.2f}")
        table.add_row("Max iterations", str(self.config.agent.max_iterations))
        table.add_row("WandB enabled", str(self.enable_wandb and self.wandb_run is not None))
        
        console.print(table)
    
    def _print_final_summary(self) -> None:
        """Print final results summary."""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.get("success", False)]
        failed = [r for r in self.results if not r.get("success", False)]
        
        avg_param_score = sum(r.get("param_score", 0) for r in self.results) / len(self.results)
        avg_vlm_score = sum(r.get("vlm_score", 0) for r in self.results) / len(self.results)
        avg_iterations = sum(r.get("iterations", 0) for r in self.results) / len(self.results)
        total_duration = sum(r.get("duration_seconds", 0) for r in self.results)
        
        table = Table(title="Processing Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total processed", str(len(self.results)))
        table.add_row("Successful", f"{len(successful)} ({100*len(successful)/len(self.results):.1f}%)")
        table.add_row("Failed", f"{len(failed)} ({100*len(failed)/len(self.results):.1f}%)")
        table.add_row("Avg param score", f"{avg_param_score:.3f}")
        table.add_row("Avg VLM score", f"{avg_vlm_score:.3f}")
        table.add_row("Avg iterations", f"{avg_iterations:.1f}")
        table.add_row("Total time", f"{total_duration:.1f}s")
        table.add_row("Avg time/sample", f"{total_duration/len(self.results):.1f}s")
        
        console.print(table)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process LLM4CAD samples through design agent")
    parser.add_argument("--data-path", type=str, default="data/LLM4CAD",
                        help="Path to LLM4CAD data")
    parser.add_argument("--categories", type=str, nargs="+",
                        help="Categories to process (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle samples")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already processed samples")
    parser.add_argument("--enable-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to SQLite database")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to checkpoint file")
    
    args = parser.parse_args()
    
    runner = DatasetRunner(
        data_path=args.data_path,
        categories=args.categories,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        checkpoint_path=args.checkpoint_path,
        enable_wandb=args.enable_wandb,
        db_path=args.db_path,
    )
    
    results = runner.run(resume=not args.no_resume)
    
    # Print database stats
    db = get_database()
    stats = db.get_stats()
    console.print(f"\n[bold]Database Stats:[/bold] {stats}")


if __name__ == "__main__":
    main()

