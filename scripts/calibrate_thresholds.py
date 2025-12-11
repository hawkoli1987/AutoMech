#!/usr/bin/env python3
"""
Threshold Calibration Script

Runs the design agent on multiple samples to collect score distributions.
Use results to calibrate param_score_threshold and vlm_score_threshold.

Run from project root: python3 scripts/calibrate_thresholds.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_servers():
    """Check if LLM and VLM servers are available."""
    import requests
    
    llm_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    vlm_base = os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
    
    llm_ok = False
    vlm_ok = False
    
    try:
        resp = requests.get(f"{llm_base}/v1/models", timeout=2)
        llm_ok = resp.status_code == 200
    except Exception:
        pass
    
    try:
        resp = requests.get(f"{vlm_base}/v1/models", timeout=2)
        vlm_ok = resp.status_code == 200
    except Exception:
        pass
    
    return llm_ok, vlm_ok


def main():
    console.print(Panel.fit("[bold cyan]Threshold Calibration[/bold cyan]"))
    console.print("[dim]Run agent on samples to collect score distributions[/dim]\n")
    
    # Check servers
    llm_ok, vlm_ok = check_servers()
    console.print(f"LLM Server (8001): {'[green]✓[/green]' if llm_ok else '[red]✗[/red]'}")
    console.print(f"VLM Server (8002): {'[green]✓[/green]' if vlm_ok else '[red]✗[/red]'}")
    
    if not llm_ok:
        console.print("[red]LLM server required. Set OPENAI_API_BASE.[/red]")
        return 1
    
    if not vlm_ok:
        console.print("[yellow]⚠ VLM server not available - VLM judge will be skipped[/yellow]")
    
    from src.data_loop.loader import LLM4CADLoader
    from src.agents.design_agent import DesignAgent
    from src.schemas import CADCategory
    from src.config import get_config
    
    config = get_config()
    data_path = os.getenv("LLM4CAD_PATH", config.data.llm4cad_path)
    
    # Load a few samples from selected categories
    categories = ["Flange", "Gear", "Nut"]  # Fast categories
    max_per_category = 2
    
    console.print(f"\n[bold]Loading samples from: {data_path}[/bold]")
    
    results = []
    param_scores = []
    vlm_scores = []
    iterations_list = []
    
    for cat_name in categories:
        category = CADCategory(cat_name)
        
        # Create loader for single category with limit
        loader = LLM4CADLoader(
            data_path=data_path,
            categories=[category],
            max_samples_per_category=max_per_category,
        )
        
        samples = list(loader)
        console.print(f"\n[cyan]{cat_name}[/cyan]: Found {len(samples)} samples")
        
        for i, sample in enumerate(samples):
            console.print(f"  Processing sample {i+1}/{len(samples)}: {sample.sample_id}...", end=" ")
            
            try:
                agent = DesignAgent()
                initial_state = sample.to_graph_state()
                final_state = agent.run(initial_state)
                
                results.append(final_state)
                
                ps = final_state.param_score
                vs = final_state.vlm_score
                it = final_state.iteration
                
                if ps is not None:
                    param_scores.append(ps)
                if vs is not None:
                    vlm_scores.append(vs)
                iterations_list.append(it)
                
                ps_str = f"{ps:.2f}" if ps is not None else "?"
                vs_str = f"{vs:.2f}" if vs is not None else "?"
                console.print(f"[green]✓[/green] param={ps_str}, vlm={vs_str}, iter={it}")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                results.append(None)
    
    # Summarize metrics
    console.print("\n[bold]Score Distributions[/bold]\n")
    
    def stats(values):
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        values = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": values[len(values) // 2],
        }
    
    ps_stats = stats(param_scores)
    vs_stats = stats(vlm_scores)
    it_stats = stats(iterations_list)
    
    # Display table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Mean")
    table.add_column("Median")
    
    for name, s in [
        ("Param Score", ps_stats),
        ("VLM Score", vs_stats),
        ("Iterations", it_stats),
    ]:
        table.add_row(
            name,
            f"{s['min']:.2f}",
            f"{s['max']:.2f}",
            f"{s['mean']:.2f}",
            f"{s['median']:.2f}",
        )
    
    console.print(table)
    
    # Raw scores
    console.print(f"\n[dim]Raw param scores: {[f'{x:.2f}' for x in param_scores]}[/dim]")
    console.print(f"[dim]Raw VLM scores: {[f'{x:.2f}' for x in vlm_scores]}[/dim]")
    
    # Recommendations
    console.print("\n[bold]Threshold Recommendations[/bold]")
    
    # Recommend threshold at ~75th percentile or median + 0.05
    if ps_stats["mean"] > 0:
        param_threshold = min(0.90, ps_stats["median"] + 0.05)
    else:
        param_threshold = 0.75
        
    if vs_stats["mean"] > 0:
        vlm_threshold = min(0.85, vs_stats["median"] + 0.05)
    else:
        vlm_threshold = 0.70
    
    console.print(f"  param_score_threshold: {param_threshold:.2f} (current: 0.85)")
    console.print(f"  vlm_score_threshold: {vlm_threshold:.2f} (current: 0.80)")
    
    if param_threshold < 0.85:
        console.print(f"\n[yellow]⚠ Consider lowering param_score_threshold to {param_threshold:.2f}[/yellow]")
    
    if vlm_threshold < 0.80:
        console.print(f"[yellow]⚠ Consider lowering vlm_score_threshold to {vlm_threshold:.2f}[/yellow]")
    
    console.print("\n[green]✓ Calibration complete![/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
