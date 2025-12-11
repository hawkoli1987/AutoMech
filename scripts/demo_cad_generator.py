#!/usr/bin/env python3
"""
Demo: CAD Generation Pipeline

End-to-end demo: Text → LLM → Parametric Spec → CadQuery/FreeCAD → CAD (STEP/STL + PNG)

This demonstrates the full CAD generation capability:
1. User provides natural language description
2. LLM extracts parametric design specification
3. CadQuery generates 3D CAD model (STEP/STL)
4. Render engine creates PNG preview

Run from project root: python3 scripts/demo_cad_generator.py

Requirements:
- LLM server running on port 8001 (Qwen3-8B)
- CadQuery installed in the environment
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax


def check_llm_available():
    """Check if LLM server is available."""
    import requests
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]CAD Generation Pipeline Demo[/bold cyan]"))
    console.print("[dim]Text → LLM → Parametric Spec → CadQuery → CAD[/dim]\n")
    
    # Check dependencies
    console.print("[bold]1. Checking Dependencies[/bold]")
    
    # Check LLM
    if check_llm_available():
        console.print("   [green]✓[/green] LLM server available")
    else:
        console.print("   [red]✗[/red] LLM server not available")
        console.print("   [dim]Set OPENAI_API_BASE=http://localhost:8001[/dim]")
        return
    
    # Check CadQuery
    from src.cad.generators import is_cadquery_available
    if is_cadquery_available():
        console.print("   [green]✓[/green] CadQuery available")
        cadquery_available = True
    else:
        console.print("   [yellow]⚠[/yellow] CadQuery not available - will show stub paths only")
        cadquery_available = False
    
    # Example text descriptions for different categories
    examples = {
        "Flange": "A circular flange with base diameter 120mm, base height 15mm, outer raised face diameter 80mm, inner bore 40mm, and flange height 10mm.",
        "Gear": "A spur gear with module 2.5, 24 teeth, face width 15mm, and bore diameter 20mm.",
        "Nut": "A hexagonal nut with size 30mm across flats, height 12mm, and internal thread diameter 16mm.",
        "Shaft": "A stepped shaft with two sections: first section diameter 25mm and length 50mm, second section diameter 20mm and length 40mm.",
        "Spring": "A compression spring with wire diameter 3mm, coil diameter 25mm, 8 active coils, and free length 60mm.",
    }
    
    from src.utils.llm_client import get_llm_client
    from src.schemas import FlangeSpec, GearSpec, NutSpec, ShaftSpec, SpringSpec
    
    # Get LLM client
    client = get_llm_client()
    console.print(f"   LLM Model: [green]{client.model}[/green]")
    
    # Process each category
    console.print("\n[bold]2. Text → Parametric Specification (via LLM)[/bold]\n")
    
    param_results = {}
    
    for category, text_desc in examples.items():
        console.print(f"   [bold cyan]{category}[/bold cyan]")
        console.print(f"   Input: [dim]{text_desc[:60]}...[/dim]")
        
        # Build prompt based on category
        if category == "Flange":
            schema_example = '{"base_diameter": 100, "base_height": 10, "outer_diameter": 70, "inner_diameter": 30, "flange_height": 8}'
        elif category == "Gear":
            schema_example = '{"module": 2.0, "num_teeth": 20, "face_width": 10, "bore_diameter": 15}'
        elif category == "Nut":
            schema_example = '{"nut_size": 24, "nut_height": 10, "inner_diameter": 12}'
        elif category == "Shaft":
            schema_example = '[[diameter1, length1], [diameter2, length2]]'
        else:  # Spring
            schema_example = '{"wire_diameter": 2, "coil_diameter": 20, "num_coils": 6, "free_length": 50}'
        
        prompt = f"""Extract the parametric specification from this {category} description.

Description: {text_desc}

Return ONLY a JSON object like this example (no explanation):
{schema_example}"""

        try:
            result = client.generate_json(prompt, max_tokens=200)
            param_results[category] = result
            
            # Display result
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            console.print(f"   Output: [green]{result_str}[/green]")
            console.print()
            
        except Exception as e:
            console.print(f"   [red]Error: {e}[/red]")
            param_results[category] = None
            console.print()
    
    # Generate CAD from parameters
    console.print("[bold]3. Parametric Specification → CAD (via CadQuery)[/bold]\n")
    
    if not cadquery_available:
        console.print("   [yellow]CadQuery not available - showing expected outputs only[/yellow]\n")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Category")
        table.add_column("STEP File (expected)")
        table.add_column("PNG Render (expected)")
        
        for category in examples.keys():
            table.add_row(
                category,
                f"artifacts/demo/{category.lower()}_demo.step",
                f"artifacts/demo/{category.lower()}_demo.png",
            )
        
        console.print(table)
        console.print("\n   [dim]Install CadQuery to generate actual CAD files[/dim]")
    
    else:
        from src.cad.generators import generate_and_export
        
        # Create output directory
        output_dir = Path("artifacts/demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"   Output directory: {output_dir}\n")
        
        results_table = Table(show_header=True, header_style="bold")
        results_table.add_column("Category")
        results_table.add_column("Status")
        results_table.add_column("STEP File")
        results_table.add_column("PNG Render")
        results_table.add_column("Volume (mm³)")
        
        for category, params in param_results.items():
            if params is None:
                results_table.add_row(category, "[red]FAILED[/red]", "-", "-", "-")
                continue
            
            try:
                result = generate_and_export(
                    category=category,
                    params=params,
                    output_dir=str(output_dir),
                    sample_id=f"{category.lower()}_demo",
                    iteration=0,
                    export_step=True,
                    export_stl=False,
                    render_png=True,
                )
                
                if result.success:
                    step_file = Path(result.cad_file).name if result.cad_file else "-"
                    png_file = Path(result.render_image).name if result.render_image else "-"
                    volume = f"{result.volume:.1f}" if result.volume else "-"
                    results_table.add_row(category, "[green]SUCCESS[/green]", step_file, png_file, volume)
                else:
                    results_table.add_row(category, f"[red]ERROR[/red]", result.error_message[:30], "-", "-")
                    
            except Exception as e:
                results_table.add_row(category, "[red]EXCEPTION[/red]", str(e)[:30], "-", "-")
        
        console.print(results_table)
        
        # Show generated files
        console.print("\n[bold]Generated Files:[/bold]")
        for f in sorted(output_dir.glob("**/*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                console.print(f"   {f.relative_to(output_dir)} ({size_kb:.1f} KB)")
    
    # Show the full pipeline summary
    console.print("\n[bold]4. Pipeline Summary[/bold]")
    
    pipeline = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CAD Generation Pipeline                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   [Text Description]                                             │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────────┐                                            │
    │   │  LLM (Qwen3-8B) │  ← Extract structured parameters           │
    │   └────────┬────────┘                                            │
    │            │                                                     │
    │            ▼                                                     │
    │   [Parametric Spec JSON]                                         │
    │   {"base_diameter": 120, "inner_diameter": 40, ...}              │
    │            │                                                     │
    │            ▼                                                     │
    │   ┌─────────────────┐                                            │
    │   │    CadQuery     │  ← Generate 3D geometry                    │
    │   └────────┬────────┘                                            │
    │            │                                                     │
    │            ├──────────────┬──────────────┐                       │
    │            ▼              ▼              ▼                       │
    │      [STEP File]    [STL File]    [PNG Render]                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """
    console.print(pipeline)
    
    console.print("[green]✓ CAD generation demo complete![/green]")
    
    if cadquery_available:
        console.print(f"\n[dim]Generated files are in: artifacts/demo/[/dim]")


if __name__ == "__main__":
    main()

