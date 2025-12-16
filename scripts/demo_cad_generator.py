#!/usr/bin/env python3
"""
Demo: CAD Generation Pipeline (Interactive)

End-to-end demo: Text → LLM → Parametric Spec → CadQuery → CAD (STEP/STL + PNG)

This demonstrates the full CAD generation capability:
1. User provides natural language description
2. LLM extracts parametric design specification
3. CadQuery generates 3D CAD model (STEP/STL)
4. Render engine creates PNG preview

Run from project root: python3 scripts/demo_cad_generator.py

Options:
  --text "description"   Custom text description
  --category CATEGORY    Specify category (Flange/Gear/Nut/Shaft/Spring)
  --examples             Run with built-in examples instead of interactive mode

Requirements:
- LLM server running on port 8001 (Qwen3-8B)
- CadQuery installed in the environment
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm


def check_llm_available():
    """Check if LLM server is available."""
    import requests
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def check_cadquery_available():
    """Check if CadQuery is available."""
    try:
        import cadquery
        return True
    except ImportError:
        return False


def detect_category(text: str) -> str:
    """Auto-detect category from text description."""
    text_lower = text.lower()
    
    if "flange" in text_lower:
        return "Flange"
    elif "gear" in text_lower or "teeth" in text_lower or "tooth" in text_lower:
        return "Gear"
    elif "nut" in text_lower or "hexagonal" in text_lower:
        return "Nut"
    elif "shaft" in text_lower or "stepped" in text_lower:
        return "Shaft"
    elif "spring" in text_lower or "coil" in text_lower:
        return "Spring"
    else:
        return "Flange"  # Default


def get_schema_example(category: str) -> str:
    """Get example JSON schema for a category."""
    examples = {
        "Flange": '{"base_diameter": 100, "base_height": 10, "outer_diameter": 70, "inner_diameter": 30, "flange_height": 8}',
        "Gear": '{"module": 2.0, "num_teeth": 20, "face_width": 10, "bore_diameter": 15}',
        "Nut": '{"nut_size": 24, "nut_height": 10, "inner_diameter": 12}',
        "Shaft": '[[25, 50], [20, 40]]',  # List of [diameter, length] pairs
        "Spring": '{"wire_diameter": 2, "coil_diameter": 20, "num_coils": 6, "free_length": 50}',
    }
    return examples.get(category, examples["Flange"])


def extract_params_from_text(client, text: str, category: str, console: Console) -> dict | list | None:
    """Use LLM to extract parametric specification from text."""
    schema_example = get_schema_example(category)
    
    prompt = f"""Extract the parametric specification from this {category} description.

Description: {text}

Return ONLY a JSON object like this example (with actual numeric values, no explanation):
{schema_example}"""

    console.print(f"\n[dim]Sending to LLM: {client.model}...[/dim]")
    
    try:
        result = client.generate_json(prompt, max_tokens=200)
        return result
    except Exception as e:
        console.print(f"[red]LLM Error: {e}[/red]")
        return None


def generate_cad_from_params(category: str, params: dict | list, output_dir: Path, console: Console):
    """Generate CAD file from parameters using CadQuery."""
    from src.cad.generators import generate_and_export
    
    sample_id = f"{category.lower()}_user"
    
    console.print(f"\n[bold]Generating CAD with CadQuery...[/bold]")
    
    result = generate_and_export(
        category=category,
        params=params,
        output_dir=str(output_dir),
        sample_id=sample_id,
        iteration=0,
        export_step=True,
        export_stl=True,
        render_png=True,
    )
    
    return result


def run_interactive(console: Console, client):
    """Run interactive mode - user enters text description."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           INTERACTIVE CAD GENERATION MODE[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print("Supported components: [green]Flange, Gear, Nut, Shaft, Spring[/green]\n")
    console.print("[dim]Example inputs:[/dim]")
    console.print("[dim]  • A circular flange with base diameter 120mm, inner bore 40mm[/dim]")
    console.print("[dim]  • A spur gear with 24 teeth, module 2.5, face width 15mm[/dim]")
    console.print("[dim]  • A hexagonal nut with size 30mm, height 12mm, thread diameter 16mm[/dim]")
    console.print("[dim]  • A stepped shaft: section 1 is 25mm × 50mm, section 2 is 20mm × 40mm[/dim]")
    console.print("[dim]  • A compression spring with wire 3mm, coil diameter 25mm, 8 coils[/dim]\n")
    
    # Get user input
    text = Prompt.ask("[bold]Enter your component description[/bold]")
    
    if not text.strip():
        console.print("[red]No description provided. Exiting.[/red]")
        return
    
    # Auto-detect category
    detected_category = detect_category(text)
    console.print(f"\n[dim]Auto-detected category: [cyan]{detected_category}[/cyan][/dim]")
    
    # Allow user to override
    override = Prompt.ask(
        "Category",
        choices=["Flange", "Gear", "Nut", "Shaft", "Spring"],
        default=detected_category
    )
    category = override
    
    # Step 1: Extract parameters
    console.print(f"\n[bold]Step 1: Text → Parametric Specification[/bold]")
    console.print(f"Input: [cyan]{text}[/cyan]")
    
    params = extract_params_from_text(client, text, category, console)
    
    if params is None:
        console.print("[red]Failed to extract parameters. Exiting.[/red]")
        return
    
    # Display extracted parameters
    params_str = json.dumps(params, indent=2) if isinstance(params, dict) else json.dumps(params)
    console.print(f"\n[green]Extracted Parameters:[/green]")
    console.print(Panel(params_str, title="Parametric Specification", border_style="green"))
    
    # Confirm before generating
    if not Confirm.ask("\nProceed with CAD generation?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    
    # Step 2: Generate CAD
    console.print(f"\n[bold]Step 2: Parametric Specification → CAD[/bold]")
    
    output_dir = Path("artifacts/user_generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = generate_cad_from_params(category, params, output_dir, console)
    
    # Display result
    if result.success:
        console.print(f"\n[green]✓ CAD generation successful![/green]\n")
        
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Property")
        table.add_column("Value")
        
        if result.cad_file:
            table.add_row("STEP File", result.cad_file)
        if result.stl_file:
            table.add_row("STL File", result.stl_file)
        if result.render_image:
            table.add_row("PNG Render", result.render_image)
        if result.volume:
            table.add_row("Volume", f"{result.volume:.2f} mm³")
        if result.bounding_box:
            bbox = result.bounding_box
            table.add_row("Bounding Box", f"{bbox['x']:.1f} × {bbox['y']:.1f} × {bbox['z']:.1f} mm")
        
        console.print(table)
        
        console.print(f"\n[dim]Files saved to: {output_dir}/[/dim]")
        
    else:
        console.print(f"\n[red]✗ CAD generation failed![/red]")
        console.print(f"Error: {result.error_message}")


def run_examples(console: Console, client):
    """Run with built-in examples for all categories."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           RUNNING BUILT-IN EXAMPLES[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    examples = {
        "Flange": "A circular flange with base diameter 120mm, base height 15mm, outer raised face diameter 80mm, inner bore 40mm, and flange height 10mm.",
        "Gear": "A spur gear with module 2.5, 24 teeth, face width 15mm, and bore diameter 20mm.",
        "Nut": "A hexagonal nut with size 30mm across flats, height 12mm, and internal thread diameter 16mm.",
        "Shaft": "A stepped shaft with two sections: first section diameter 25mm and length 50mm, second section diameter 20mm and length 40mm.",
        "Spring": "A compression spring with wire diameter 3mm, coil diameter 25mm, 8 active coils, and free length 60mm.",
    }
    
    output_dir = Path("artifacts/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_table = Table(show_header=True, header_style="bold")
    results_table.add_column("Category")
    results_table.add_column("Status")
    results_table.add_column("STEP File")
    results_table.add_column("Volume (mm³)")
    
    for category, text_desc in examples.items():
        console.print(f"\n[bold cyan]Processing: {category}[/bold cyan]")
        console.print(f"[dim]{text_desc[:70]}...[/dim]")
        
        # Extract parameters
        params = extract_params_from_text(client, text_desc, category, console)
        
        if params is None:
            results_table.add_row(category, "[red]LLM FAILED[/red]", "-", "-")
            continue
        
        console.print(f"[green]Params: {json.dumps(params) if isinstance(params, dict) else params}[/green]")
        
        # Generate CAD
        from src.cad.generators import generate_and_export
        
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
                volume = f"{result.volume:.1f}" if result.volume else "-"
                results_table.add_row(category, "[green]SUCCESS[/green]", step_file, volume)
            else:
                results_table.add_row(category, "[red]CAD FAILED[/red]", result.error_message[:30], "-")
                
        except Exception as e:
            results_table.add_row(category, "[red]EXCEPTION[/red]", str(e)[:30], "-")
    
    console.print("\n[bold]Results Summary:[/bold]\n")
    console.print(results_table)
    
    # List generated files
    console.print("\n[bold]Generated Files:[/bold]")
    for f in sorted(output_dir.glob("**/*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            console.print(f"   {f.relative_to(output_dir)} ({size_kb:.1f} KB)")
    
    console.print(f"\n[green]✓ Examples complete! Files saved to: {output_dir}/[/green]")


def main():
    parser = argparse.ArgumentParser(description="CAD Generation Pipeline Demo")
    parser.add_argument("--text", type=str, help="Custom text description for CAD generation")
    parser.add_argument("--category", type=str, choices=["Flange", "Gear", "Nut", "Shaft", "Spring"],
                        help="Specify component category")
    parser.add_argument("--examples", action="store_true", help="Run with built-in examples")
    args = parser.parse_args()
    
    console = Console()
    console.print(Panel.fit("[bold cyan]CAD Generation Pipeline Demo[/bold cyan]"))
    console.print("[dim]Text → LLM → Parametric Spec → CadQuery → CAD[/dim]\n")
    
    # Check dependencies
    console.print("[bold]Checking Dependencies...[/bold]")
    
    # Check LLM
    if not check_llm_available():
        console.print("[red]✗ LLM server not available[/red]")
        console.print("[dim]Start LLM server: export OPENAI_API_BASE=http://localhost:8001[/dim]")
        return 1
    console.print("[green]✓[/green] LLM server available")
    
    # Check CadQuery
    if not check_cadquery_available():
        console.print("[red]✗ CadQuery not installed[/red]")
        console.print("[dim]Install: pip install cadquery[/dim]")
        return 1
    console.print("[green]✓[/green] CadQuery available")
    
    # Get LLM client
    from src.utils.llm_client import get_llm_client
    client = get_llm_client()
    console.print(f"[green]✓[/green] LLM Model: {client.model}")
    
    # Run mode
    if args.examples:
        run_examples(console, client)
    elif args.text:
        # Direct mode with command-line text
        category = args.category or detect_category(args.text)
        console.print(f"\n[bold]Processing: {category}[/bold]")
        console.print(f"Input: [cyan]{args.text}[/cyan]")
        
        params = extract_params_from_text(client, args.text, category, console)
        if params:
            console.print(f"[green]Params: {json.dumps(params) if isinstance(params, dict) else params}[/green]")
            
            output_dir = Path("artifacts/user_generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = generate_cad_from_params(category, params, output_dir, console)
            if result.success:
                console.print(f"\n[green]✓ Success! Files saved to {output_dir}/[/green]")
                if result.cad_file:
                    console.print(f"   STEP: {result.cad_file}")
                if result.stl_file:
                    console.print(f"   STL: {result.stl_file}")
            else:
                console.print(f"[red]✗ Failed: {result.error_message}[/red]")
    else:
        # Interactive mode
        run_interactive(console, client)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
