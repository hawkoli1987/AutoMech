#!/usr/bin/env python3
"""
Demo: VLM Judge Integration

Tests the VLM judge with actual CAD renders from the LLM4CAD dataset.
Shows how the VLM evaluates geometric accuracy and provides feedback.

Run from project root: python3 scripts/demo_vlm_judge.py
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]VLM Judge Integration Demo[/bold cyan]"))
    
    # Check VLM server
    api_base = os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
    console.print(f"\n[dim]VLM API: {api_base}[/dim]")
    
    import requests
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        if response.status_code != 200:
            console.print("[red]VLM server not available[/red]")
            return
    except Exception as e:
        console.print(f"[red]VLM server not available: {e}[/red]")
        return
    
    from src.utils.llm_client import get_vlm_client
    from src.data_loop.loader import LLM4CADLoader
    from src.schemas import CADCategory
    from src.agents.design_agent import JUDGE_VLM_SYSTEM, JUDGE_VLM_USER
    import json
    
    # Initialize VLM client
    console.print("\n[bold]1. Initialize VLM Client[/bold]")
    vlm_client = get_vlm_client()
    console.print(f"   Model: [green]{vlm_client.model}[/green]")
    
    # Load a sample with image
    console.print("\n[bold]2. Load Sample with Image[/bold]")
    
    data_path = Path("data/LLM4CAD")
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data_path}[/red]")
        return
    
    # Find samples with images
    loader = LLM4CADLoader(data_path=data_path, max_samples_per_category=5)
    
    # Find a sample with an existing image
    test_sample = None
    test_image = None
    for sample in loader:
        # Check for image in img/ folder
        img_path = data_path / sample.category.value / "img" / f"{sample.sample_id}.png"
        if img_path.exists():
            test_sample = sample
            test_image = str(img_path)
            break
    
    if not test_sample or not test_image:
        console.print("[yellow]No samples with images found[/yellow]")
        # Try to find any image
        images = glob.glob(str(data_path / "**/img/*.png"), recursive=True)
        if images:
            test_image = images[0]
            console.print(f"   Using fallback image: {test_image}")
        else:
            console.print("[red]No images available for testing[/red]")
            return
    else:
        console.print(f"   Sample: {test_sample.sample_id}")
        console.print(f"   Category: {test_sample.category.value}")
        console.print(f"   Image: {test_image}")
    
    # Test VLM evaluation
    console.print("\n[bold]3. VLM Evaluation[/bold]")
    
    # Use the sample's parameters or mock parameters
    if test_sample:
        pred_spec = test_sample.gt_param_spec
        text_desc = test_sample.text_desc
        category = test_sample.category.value
    else:
        pred_spec = {"outer_diameter": 100, "inner_diameter": 50, "height": 20}
        text_desc = "A mechanical part for testing"
        category = "Flange"
    
    pred_spec_str = json.dumps(pred_spec, indent=2) if isinstance(pred_spec, dict) else str(pred_spec)
    
    prompt = JUDGE_VLM_USER.format(
        category=category,
        pred_spec=pred_spec_str,
        text_desc=text_desc,
    )
    
    console.print(f"   Sending image to VLM for evaluation...")
    console.print(f"   [dim]Category: {category}[/dim]")
    console.print(f"   [dim]Parameters: {pred_spec_str[:100]}...[/dim]")
    
    try:
        result = vlm_client.generate_json_with_image(
            prompt=prompt,
            image_path=test_image,
            system_prompt=JUDGE_VLM_SYSTEM,
            temperature=0.2,
            max_tokens=600,
        )
        
        console.print("\n   [green]VLM Response:[/green]")
        
        # Display scores in a table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Score Type")
        table.add_column("Value")
        table.add_column("Threshold")
        table.add_column("Status")
        
        def score_status(score, threshold=0.8):
            return "[green]PASS[/green]" if score >= threshold else "[red]FAIL[/red]"
        
        overall = float(result.get("overall_score", 0.5))
        geometric = float(result.get("geometric_score", 0.5))
        completeness = float(result.get("completeness_score", 0.5))
        quality = float(result.get("quality_score", 0.5))
        
        table.add_row("Overall", f"{overall:.3f}", "0.80", score_status(overall))
        table.add_row("Geometric", f"{geometric:.3f}", "0.80", score_status(geometric))
        table.add_row("Completeness", f"{completeness:.3f}", "0.80", score_status(completeness))
        table.add_row("Quality", f"{quality:.3f}", "0.80", score_status(quality))
        
        console.print(table)
        
        # Show feedback
        feedback = result.get("feedback", "No feedback")
        console.print(f"\n   [bold]Feedback:[/bold] {feedback}")
        
        # Show issues if any
        issues = result.get("issues", [])
        if issues:
            console.print(f"\n   [bold]Issues Found:[/bold]")
            for issue in issues:
                console.print(f"     • {issue}")
        
    except Exception as e:
        console.print(f"   [red]VLM evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    # Test multiple samples
    console.print("\n[bold]4. Batch Evaluation (3 samples)[/bold]")
    
    results = []
    count = 0
    for sample in loader:
        if count >= 3:
            break
        
        img_path = data_path / sample.category.value / "img" / f"{sample.sample_id}.png"
        if not img_path.exists():
            continue
        
        pred_spec_str = json.dumps(sample.gt_param_spec, indent=2) if isinstance(sample.gt_param_spec, dict) else str(sample.gt_param_spec)
        
        prompt = JUDGE_VLM_USER.format(
            category=sample.category.value,
            pred_spec=pred_spec_str,
            text_desc=sample.text_desc[:200],
        )
        
        try:
            result = vlm_client.generate_json_with_image(
                prompt=prompt,
                image_path=str(img_path),
                system_prompt=JUDGE_VLM_SYSTEM,
                temperature=0.2,
                max_tokens=600,
            )
            
            results.append({
                "sample_id": sample.sample_id,
                "category": sample.category.value,
                "overall": float(result.get("overall_score", 0.5)),
                "geometric": float(result.get("geometric_score", 0.5)),
            })
            count += 1
            console.print(f"   Evaluated: {sample.sample_id} → {results[-1]['overall']:.3f}")
            
        except Exception as e:
            console.print(f"   [yellow]Failed: {sample.sample_id} - {e}[/yellow]")
    
    if results:
        console.print("\n   [bold]Summary:[/bold]")
        avg_overall = sum(r["overall"] for r in results) / len(results)
        avg_geometric = sum(r["geometric"] for r in results) / len(results)
        console.print(f"   Average Overall Score: {avg_overall:.3f}")
        console.print(f"   Average Geometric Score: {avg_geometric:.3f}")
    
    console.print("\n[green]✓ VLM judge demo complete![/green]")


if __name__ == "__main__":
    main()

