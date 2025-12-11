#!/usr/bin/env python3
"""
Demo: LLM4CAD Data Loader

Shows how to load samples from the LLM4CAD dataset.
Run from project root: python3 scripts/demo_data_loader.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data_loop.loader import LLM4CADLoader, load_llm4cad_samples
from src.schemas import CADCategory


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]LLM4CAD Data Loader Demo[/bold cyan]"))
    
    # Initialize loader
    data_path = Path("data/LLM4CAD")
    if not data_path.exists():
        console.print(f"[red]Error: Data directory not found: {data_path}[/red]")
        console.print("Make sure you're running from the project root.")
        return
    
    loader = LLM4CADLoader(data_path=data_path, max_samples_per_category=5)
    
    # Show category counts
    console.print("\n[bold]Category Counts:[/bold]")
    counts = loader.get_category_counts()
    for cat, count in counts.items():
        console.print(f"  {cat}: {count} samples")
    
    # Load and display samples by iterating
    console.print("\n[bold]Sample Preview:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Sample ID", width=18)
    table.add_column("Category", width=10)
    table.add_column("Text Description", width=45)
    table.add_column("Params", width=25)
    
    for i, sample in enumerate(loader):
        if i >= 15:  # Show first 15 samples
            break
            
        # Truncate text description
        text = sample.text_desc[:60] + "..." if len(sample.text_desc) > 60 else sample.text_desc
        
        # Format params
        if isinstance(sample.gt_param_spec, list):
            params = f"Sections: {len(sample.gt_param_spec)} steps"
        else:
            keys = list(sample.gt_param_spec.keys())[:3]
            params = ", ".join(f"{k}={sample.gt_param_spec[k]}" for k in keys)
            if len(sample.gt_param_spec) > 3:
                params += "..."
        
        table.add_row(sample.sample_id, sample.category.value, text, params)
    
    console.print(table)
    
    # Load a specific sample
    console.print("\n[bold]Load Specific Sample:[/bold]")
    sample = loader.load_sample("Flange_00001", CADCategory.FLANGE)
    if sample:
        console.print(f"  ID: {sample.sample_id}")
        console.print(f"  Category: {sample.category.value}")
        console.print(f"  Text: {sample.text_desc[:80]}...")
        console.print(f"  STL Path: {sample.stl_path}")
    else:
        console.print("  [yellow]Sample not found[/yellow]")
    
    # Show GraphState conversion
    console.print("\n[bold]GraphState Conversion:[/bold]")
    sample = next(iter(loader))
    state = sample.to_graph_state()
    
    console.print(f"  text_desc: '{state.text_desc[:50]}...'")
    console.print(f"  gt_param_spec type: {type(state.gt_param_spec).__name__}")
    console.print(f"  iteration: {state.iteration}")
    console.print(f"  done: {state.done}")
    console.print(f"  metadata: sample_id={state.metadata.sample_id}, category={state.metadata.category}")
    
    # Convenience function demo
    console.print("\n[bold]Convenience Function:[/bold]")
    samples = load_llm4cad_samples(max_samples=5)
    console.print(f"  Loaded {len(samples)} samples using load_llm4cad_samples()")
    
    console.print("\n[green]✓ Data loader demo complete![/green]")


if __name__ == "__main__":
    main()
