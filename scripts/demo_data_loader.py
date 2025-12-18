#!/usr/bin/env python3
"""
Demo: Simple Text Sample Creation

Shows how to create samples for freeform CAD generation from text descriptions.
Run from project root: python3 scripts/demo_data_loader.py

Note: The category-specific LLM4CAD data loader has been removed.
This demo now focuses on freeform CAD generation from text.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data_loop.loader import create_sample_from_text


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Freeform CAD Sample Creation Demo[/bold cyan]"))
    
    # Example text descriptions for various mechanical parts
    example_descriptions = [
        "A mounting bracket with two 8mm holes spaced 40mm apart",
        "A cylindrical spacer with outer diameter 30mm, inner diameter 20mm, and height 15mm",
        "An L-shaped bracket with 50mm × 30mm dimensions and 5mm thickness",
        "A T-joint connector with 20mm diameter arms",
        "A custom flange with base diameter 120mm and inner bore 40mm",
    ]
    
    console.print("\n[bold]Creating Samples from Text Descriptions:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Sample ID", width=20)
    table.add_column("Text Description", width=60)
    
    samples = []
    for desc in example_descriptions:
        sample = create_sample_from_text(desc)
        samples.append(sample)
        
        # Truncate text for display
        display_text = desc[:70] + "..." if len(desc) > 70 else desc
        table.add_row(sample.sample_id, display_text)
    
    console.print(table)
    
    # Show sample details
    console.print("\n[bold]Sample Details (First Sample):[/bold]")
    first_sample = samples[0]
    console.print(f"  ID: {first_sample.sample_id}")
    console.print(f"  Text: {first_sample.text_desc}")
    console.print(f"  GT Params: {first_sample.gt_param_spec}")
    console.print(f"  STL Path: {first_sample.stl_path}")
    
    # Convert to GraphState
    console.print("\n[bold]Convert to GraphState:[/bold]")
    graph_state = first_sample.to_graph_state(run_id="demo_run_001")
    console.print(f"  Text: {graph_state.text_desc}")
    console.print(f"  Sample ID: {graph_state.metadata.sample_id}")
    console.print(f"  Run ID: {graph_state.metadata.run_id}")
    console.print(f"  Iteration: {graph_state.iteration}")
    console.print(f"  Done: {graph_state.done}")
    
    console.print("\n[green]✓ Sample creation demo complete![/green]")
    console.print("\n[dim]For freeform CAD generation, use these samples with the design agent")
    console.print("to generate CadQuery code and create arbitrary mechanical parts.[/dim]\n")


if __name__ == "__main__":
    main()
