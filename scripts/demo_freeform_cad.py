#!/usr/bin/env python3
"""
Demo: Freeform CAD Code Generation (Proposal 2)

This demo showcases the freeform code generation capability where the LLM
generates complete CadQuery Python code for arbitrary mechanical parts,
beyond fixed templates.

Flow:
    Text Description → LLM → CadQuery Code → Safe Execution → CAD Model + Render

Usage:
    python3 scripts/demo_freeform_cad.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich import box
import json

from src.utils.llm_client import get_llm_client
from src.schemas import CadQueryCodeDesign
from src.cad.code_executor import execute_cadquery_code, preview_code_safety
from src.cad.generators import generate_from_code
from src.config import get_config


# =============================================================================
# Code Generation Prompts (from design_agent.py)
# =============================================================================

CODEGEN_SYSTEM = """You are an expert CadQuery programmer specializing in parametric 3D CAD modeling.

Generate Python code using the CadQuery API to create mechanical parts from text descriptions.

CODE TEMPLATE:
```python
# NOTE: Do NOT include any import statements!
# The 'cq' module is pre-injected and already available.

# 1. Start with a workplane
result = cq.Workplane("XY")

# 2. Build base geometry (use .circle().extrude() for cylinders, NOT .cylinder())
result = result.circle(radius).extrude(height)
# OR for rectangular parts
result = result.box(length, width, height)

# 3. Add features (holes, cutouts, etc.)
result = result.faces(">Z").workplane().circle(hole_radius).cutThruAll()

# 4. Boolean operations (if needed)
other_part = cq.Workplane("XY").circle(r).extrude(h)
result = result.union(other_part)  # or .cut() or .intersect()

# 5. Finishing touches
result = result.edges("|Z").fillet(radius)
```

CRITICAL RULES:
1. **DO NOT include any 'import' statements** - cq is already available
2. **Always assign the final shape to variable 'result'**
3. **Use .circle().extrude() for cylinders** - NOT .cylinder() which creates Compound geometry
4. **All dimensions in millimeters**
5. Comment each logical step clearly
6. Use descriptive intermediate variables when helpful
7. No external file I/O operations
8. No infinite loops or recursion
9. Infer missing dimensions using standard engineering practices

COMMON PATTERNS:
- Cylinders: cq.Workplane("XY").circle(radius).extrude(height)
- Holes: .faces(">Z").workplane().pushPoints([...]).circle(r).cutThruAll()
- Chamfers: .edges().chamfer(distance)
- Fillets: .edges().fillet(radius)
- Arrays: .rarray(xSpacing, ySpacing, xCount, yCount)
- Selection: .faces(">Z") (top face), .edges("|Z") (vertical edges)

OUTPUT FORMAT:
{
    "description": "Brief summary of the design",
    "code": "# cq is already available\\nresult = cq.Workplane('XY')...",
    "entry_point": "result",
    "required_imports": [],
    "comments": "Any design notes or assumptions"
}

**IMPORTANT**: Do NOT include "import cadquery as cq" or any import statements in the code field!

Output only valid JSON matching this schema."""

CODEGEN_USER = """Generate CadQuery code to create this mechanical part:

Description:
{text_desc}

Think step-by-step:
1. What base shape do I need? (box, cylinder, sphere)
2. What dimensions should I use? (infer from description)
3. What features do I need to add? (holes, chamfers, fillets)
4. What boolean operations? (cut, union, intersect)
5. How do I select the right faces/edges?

Output only valid JSON with the 'description', 'code', 'entry_point', 'required_imports', and 'comments' fields."""


# =============================================================================
# Demo Functions
# =============================================================================

def generate_code_from_text(console: Console, text_desc: str) -> dict:
    """
    Step 1: Generate CadQuery code from text description.
    
    Returns:
        Dictionary with code_design fields
    """
    console.print(f"\n[bold cyan]Step 1: Generating CadQuery Code[/bold cyan]")
    console.print(f"[dim]Text: {text_desc}[/dim]\n")
    
    client = get_llm_client()
    
    prompt = CODEGEN_USER.format(text_desc=text_desc)
    
    try:
        console.print("[dim]→ Calling LLM to generate code...[/dim]")
        code_dict = client.generate_json(
            prompt=prompt,
            system_prompt=CODEGEN_SYSTEM,
            temperature=0.3,
            max_tokens=2048,
        )
        
        # Validate required fields
        if not isinstance(code_dict, dict) or "code" not in code_dict:
            console.print("[red]✗ LLM did not return valid code structure[/red]")
            return None
        
        # Set defaults
        code_dict.setdefault("description", text_desc[:100])
        code_dict.setdefault("entry_point", "result")
        code_dict.setdefault("required_imports", ["cadquery as cq"])
        code_dict.setdefault("comments", "")
        
        console.print("[green]✓ Code generated successfully[/green]")
        return code_dict
        
    except Exception as e:
        console.print(f"[red]✗ Code generation failed: {e}[/red]")
        return None


def validate_and_display_code(console: Console, code_dict: dict) -> bool:
    """
    Step 2: Validate code safety and display it.
    
    Returns:
        True if code passes all safety checks
    """
    console.print(f"\n[bold cyan]Step 2: Code Validation[/bold cyan]\n")
    
    code = code_dict.get("code", "")
    
    # Run safety checks
    safety_results = preview_code_safety(code)
    
    # Display results table
    table = Table(title="Safety Validation Results", box=box.ROUNDED)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    for check_name, check_result in safety_results.items():
        if check_name == "overall":
            continue
        
        passed = check_result["passed"]
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        message = check_result.get("message") or "OK"
        
        table.add_row(
            check_name.replace("_", " ").title(),
            status,
            message if not passed else "[dim]OK[/dim]"
        )
    
    console.print(table)
    
    overall_passed = safety_results["overall"]["passed"]
    
    if not overall_passed:
        console.print("\n[red]✗ Code failed safety validation[/red]")
        return False
    
    console.print("\n[green]✓ Code passed all safety checks[/green]")
    
    # Display the generated code
    console.print("\n[bold]Generated Code:[/bold]")
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="CadQuery Code", border_style="green"))
    
    if code_dict.get("comments"):
        console.print(f"\n[dim]Comments: {code_dict['comments']}[/dim]")
    
    return True


def execute_code(console: Console, code_dict: dict) -> any:
    """
    Step 3: Execute code in sandboxed environment.
    
    Returns:
        CadQuery Workplane if successful, None otherwise
    """
    console.print(f"\n[bold cyan]Step 3: Executing Code Safely[/bold cyan]\n")
    
    try:
        code_design = CadQueryCodeDesign(**code_dict)
        console.print("[dim]→ Running code in sandboxed environment...[/dim]")
        
        workplane, error = execute_cadquery_code(code_design)
        
        if workplane is None:
            console.print(f"[red]✗ Execution failed: {error}[/red]")
            return None
        
        console.print("[green]✓ Code executed successfully[/green]")
        console.print("[dim]→ Geometry created and validated[/dim]")
        return workplane
        
    except Exception as e:
        console.print(f"[red]✗ Execution error: {e}[/red]")
        return None


def export_cad(console: Console, code_dict: dict) -> tuple:
    """
    Step 4: Export to STEP and render PNG.
    
    Returns:
        (step_path, png_path) if successful, (None, None) otherwise
    """
    console.print(f"\n[bold cyan]Step 4: Exporting CAD Files[/bold cyan]\n")
    
    try:
        code_design = CadQueryCodeDesign(**code_dict)
        
        config = get_config()
        output_dir = config.outputs.cad_dir
        filename_prefix = "demo_freeform"
        
        console.print(f"[dim]→ Exporting to: {output_dir}[/dim]")
        
        result = generate_from_code(
            code_design=code_design,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )
        
        if not result.success:
            console.print(f"[red]✗ Export failed: {result.error_message}[/red]")
            return None, None
        
        console.print(f"[green]✓ STEP file: {result.cad_file}[/green]")
        if result.stl_file:
            console.print(f"[green]✓ STL file: {result.stl_file}[/green]")
        if result.render_image:
            console.print(f"[green]✓ Render PNG: {result.render_image}[/green]")
        if result.volume:
            console.print(f"[dim]  Volume: {result.volume:.2f} mm³[/dim]")
        
        return result.cad_file, result.render_image
        
    except Exception as e:
        console.print(f"[red]✗ Export error: {e}[/red]")
        return None, None


# =============================================================================
# Main Demo
# =============================================================================

def run_interactive_demo(console: Console):
    """Run interactive freeform CAD generation demo."""
    
    console.print(Panel.fit(
        "[bold white]Freeform CAD Code Generation Demo[/bold white]\n"
        "[dim]LLM generates complete CadQuery code for arbitrary parts[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    console.print("\n[bold]Example Descriptions:[/bold]")
    console.print("[dim]  • A mounting bracket with two holes and a curved support arm[/dim]")
    console.print("[dim]  • A T-joint connector with 20mm diameter arms[/dim]")
    console.print("[dim]  • An L-shaped bracket, 50mm × 30mm with 5mm thickness[/dim]")
    console.print("[dim]  • A custom spacer ring with outer diameter 40mm, inner 25mm, height 10mm[/dim]")
    console.print("[dim]  • A pyramidal part with 50mm square base and 80mm height[/dim]\n")
    
    # Get user input
    text_desc = Prompt.ask("[bold]Enter mechanical part description[/bold]")
    
    if not text_desc.strip():
        console.print("[yellow]No description provided, exiting.[/yellow]")
        return
    
    # Step 1: Generate code
    code_dict = generate_code_from_text(console, text_desc)
    if not code_dict:
        return
    
    # Step 2: Validate and display
    is_safe = validate_and_display_code(console, code_dict)
    if not is_safe:
        console.print("\n[yellow]⚠ Code failed safety checks. Stopping.[/yellow]")
        return
    
    # Ask user to confirm execution
    console.print()
    should_execute = Confirm.ask("[bold]Execute this code?[/bold]", default=True)
    if not should_execute:
        console.print("[yellow]Execution cancelled by user.[/yellow]")
        return
    
    # Step 3: Execute code
    workplane = execute_code(console, code_dict)
    if not workplane:
        return
    
    # Step 4: Export files
    step_path, png_path = export_cad(console, code_dict)
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]✓ Freeform CAD Generation Complete![/bold green]")
    console.print("="*60)
    
    if step_path:
        console.print(f"\n[cyan]STEP file:[/cyan] {step_path}")
    if png_path:
        console.print(f"[cyan]Render:[/cyan] {png_path}")
    
    console.print("\n[dim]This demonstrates how LLMs can generate arbitrary CAD models")
    console.print("beyond fixed templates, enabling truly flexible parametric design.[/dim]\n")


def run_predefined_examples(console: Console):
    """Run a few predefined examples without user interaction."""
    
    examples = [
        "A mounting bracket with two 8mm holes spaced 40mm apart",
        "A cylindrical spacer with outer diameter 30mm, inner 20mm, height 15mm",
        "A hexagonal nut with size 25mm, height 12mm, thread diameter 14mm",
    ]
    
    console.print(Panel.fit(
        "[bold white]Freeform CAD - Automated Examples[/bold white]",
        border_style="cyan"
    ))
    
    for i, text_desc in enumerate(examples, 1):
        console.print(f"\n[bold cyan]Example {i}/{len(examples)}[/bold cyan]: {text_desc}\n")
        
        code_dict = generate_code_from_text(console, text_desc)
        if code_dict:
            validate_and_display_code(console, code_dict)
            workplane = execute_code(console, code_dict)
            if workplane:
                export_cad(console, code_dict)
        
        console.print("\n" + "-"*60 + "\n")


def main():
    console = Console()
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        run_predefined_examples(console)
    else:
        run_interactive_demo(console)


if __name__ == "__main__":
    main()

