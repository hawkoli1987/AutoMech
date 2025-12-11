#!/usr/bin/env python3
"""
Demo: VLM Client

Shows how to use the VLM client for image-based generation.
Requires: OPENAI_API_BASE2=http://localhost:8002 (Qwen3-VL)

Run from project root: python3 scripts/demo_vlm_client.py
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]VLM Client Demo[/bold cyan]"))
    
    # Check environment
    api_base = os.getenv("OPENAI_API_BASE2", "http://localhost:8002")
    console.print(f"\n[dim]VLM API Base: {api_base}[/dim]")
    
    from src.utils.llm_client import VLMClient, get_vlm_client
    
    # Find a sample image
    data_root = Path("data/LLM4CAD")
    images = glob.glob(str(data_root / "**/img/*.png"), recursive=True)
    
    if not images:
        console.print("[red]No images found in data/LLM4CAD[/red]")
        console.print("Make sure the dataset is available.")
        return
    
    image_path = images[0]
    console.print(f"\n[dim]Using image: {image_path}[/dim]")
    
    # Initialize client
    console.print("\n[bold]1. Initializing VLM Client...[/bold]")
    try:
        client = get_vlm_client()
        console.print(f"   Model: [green]{client.model}[/green]")
        console.print(f"   Temperature: {client.temperature}")
    except Exception as e:
        console.print(f"[red]Error connecting to VLM server: {e}[/red]")
        console.print("Make sure Qwen3-VL is running on port 8002.")
        return
    
    # Image description
    console.print("\n[bold]2. Image Description[/bold]")
    prompt = "Describe this mechanical part in detail. What type of component is it?"
    console.print(f"   Prompt: [dim]{prompt}[/dim]")
    
    response = client.generate_with_image(
        prompt=prompt,
        image_path=image_path,
        max_tokens=200,
    )
    console.print(f"   Response: {response}")
    
    # JSON extraction from image
    console.print("\n[bold]3. Structured Analysis (JSON)[/bold]")
    prompt = '''Analyze this mechanical part image and return a JSON object with:
{
    "part_type": "name of the part type",
    "features": ["list", "of", "visible", "features"],
    "estimated_complexity": "low/medium/high"
}
Only output valid JSON.'''
    
    console.print(f"   Prompt: [dim]{prompt[:60]}...[/dim]")
    
    result = client.generate_json_with_image(
        prompt=prompt,
        image_path=image_path,
        max_tokens=200,
    )
    console.print(f"   Result: {result}")
    
    # CAD quality evaluation (simulating VLM judge)
    console.print("\n[bold]4. CAD Quality Evaluation[/bold]")
    prompt = '''You are a CAD quality judge. Evaluate this rendered mechanical part.

Score the following from 0.0 to 1.0:
- visual_quality: How well-rendered is the image?
- geometry_correctness: Does the geometry look valid?
- manufacturing_feasibility: Could this part be manufactured?

Return JSON only:
{"visual_quality": 0.0-1.0, "geometry_correctness": 0.0-1.0, "manufacturing_feasibility": 0.0-1.0, "feedback": "brief feedback"}'''
    
    console.print(f"   Evaluating CAD render...")
    
    result = client.generate_json_with_image(
        prompt=prompt,
        image_path=image_path,
        max_tokens=300,
    )
    console.print(f"   Evaluation: {result}")
    
    console.print("\n[green]✓ VLM client demo complete![/green]")


if __name__ == "__main__":
    main()

