#!/usr/bin/env python3
"""
Demo: Storage Layer

Shows how to use the SQLite storage layer.
Run from project root: python3 scripts/demo_storage.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Storage Layer Demo[/bold cyan]"))
    
    from src.storage.database import DatabaseManager, Sample, Run
    import tempfile
    import os
    
    # Create temporary database for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo.sqlite"
        console.print(f"\n[dim]Demo database: {db_path}[/dim]")
        
        # Initialize database
        console.print("\n[bold]1. Initialize Database[/bold]")
        db = DatabaseManager(db_path=db_path)
        console.print(f"   Tables created: samples, runs, node_executions, artifacts")
        
        # Insert samples
        console.print("\n[bold]2. Insert Samples[/bold]")
        
        samples = [
            {"id": "flange_00001", "category": "Flange", "text_desc": "A steel flange...", "gt_param_spec": {"diameter": 100}},
            {"id": "gear_00001", "category": "Gear", "text_desc": "A spur gear...", "gt_param_spec": {"module": 2}},
            {"id": "shaft_00001", "category": "Shaft", "text_desc": "A stepped shaft...", "gt_param_spec": [[20, 50], [15, 30]]},
        ]
        
        for s in samples:
            db.upsert_sample(s["id"], s["category"], s["text_desc"], s["gt_param_spec"])
            console.print(f"   Inserted: {s['id']}")
        
        # Create runs
        console.print("\n[bold]3. Create Runs[/bold]")
        
        run1 = db.create_run(sample_id="flange_00001")
        console.print(f"   Created run: {run1.id[:8]}... for flange_00001")
        
        run2 = db.create_run(sample_id="gear_00001")
        console.print(f"   Created run: {run2.id[:8]}... for gear_00001")
        
        # Log node executions
        console.print("\n[bold]4. Log Node Executions[/bold]")
        
        db.log_node_execution(
            run_id=run1.id,
            node_name="generate_param_spec",
            input_data={"text_desc": "A steel flange..."},
            output_data={"outer_diameter": 100, "inner_diameter": 50},
            duration_ms=1523,
            success=True,
        )
        console.print(f"   Logged: generate_param_spec (1523ms)")
        
        db.log_node_execution(
            run_id=run1.id,
            node_name="judge_param_spec",
            input_data={"pred": {"outer_diameter": 100}},
            output_data={"score": 0.95, "feedback": "Excellent match"},
            duration_ms=892,
            success=True,
        )
        console.print(f"   Logged: judge_param_spec (892ms)")
        
        # Log artifacts
        console.print("\n[bold]5. Log Artifacts[/bold]")
        
        db.log_artifact(
            run_id=run1.id,
            artifact_type="cad_step",
            file_path="artifacts/cad/flange_00001.step",
            file_size_bytes=45678,
        )
        console.print(f"   Logged: cad_step (45KB)")
        
        db.log_artifact(
            run_id=run1.id,
            artifact_type="render_png",
            file_path="artifacts/renders/flange_00001.png",
            file_size_bytes=123456,
        )
        console.print(f"   Logged: render_png (123KB)")
        
        # Complete run
        console.print("\n[bold]6. Complete Run[/bold]")
        
        db.complete_run(
            run_id=run1.id,
            success=True,
            total_iterations=2,
            final_param_score=0.95,
            final_vlm_score=0.88,
            final_pred_param_spec={"outer_diameter": 100, "inner_diameter": 50},
            final_cad_file="artifacts/cad/flange_00001.step",
            final_render_image="artifacts/renders/flange_00001.png",
        )
        console.print(f"   Run completed: success=True, param_score=0.95")
        
        # Query statistics
        console.print("\n[bold]7. Statistics[/bold]")
        
        stats = db.get_stats()
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Total Samples", str(stats["total_samples"]))
        table.add_row("Total Runs", str(stats["total_runs"]))
        table.add_row("Successful Runs", str(stats["successful_runs"]))
        table.add_row("Total Node Executions", str(stats["total_node_executions"]))
        table.add_row("Total Artifacts", str(stats["total_artifacts"]))
        
        console.print(table)
        
        # Query runs for sample
        console.print("\n[bold]8. Query Runs[/bold]")
        
        runs = db.get_runs_for_sample("flange_00001")
        for run in runs:
            console.print(f"   Run {run.id[:8]}: success={run.success}, score={run.final_param_score}")
        
    console.print("\n[green]✓ Storage demo complete![/green]")


if __name__ == "__main__":
    main()

