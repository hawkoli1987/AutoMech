#!/usr/bin/env python3
"""
Demo: LLM Client

Shows how to use the LLM client for text and structured generation.
Requires: OPENAI_API_BASE=http://localhost:8001 (Qwen3-8B)

Run from project root: python3 scripts/demo_llm_client.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pydantic import BaseModel, Field


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]LLM Client Demo[/bold cyan]"))
    
    # Check environment
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    console.print(f"\n[dim]API Base: {api_base}[/dim]")
    
    from src.utils.llm_client import LLMClient, get_llm_client
    
    # Initialize client
    console.print("\n[bold]1. Initializing LLM Client...[/bold]")
    try:
        client = get_llm_client()
        console.print(f"   Model: [green]{client.model}[/green]")
        console.print(f"   Temperature: {client.temperature}")
        console.print(f"   Max Tokens: {client.max_tokens}")
    except Exception as e:
        console.print(f"[red]Error connecting to LLM server: {e}[/red]")
        console.print("Make sure vLLM is running on the specified port.")
        return
    
    # Simple text generation
    console.print("\n[bold]2. Simple Text Generation[/bold]")
    prompt = "What is a flange in mechanical engineering? Answer in 2 sentences."
    console.print(f"   Prompt: [dim]{prompt}[/dim]")
    
    response = client.generate(prompt, max_tokens=100)
    console.print(f"   Response: {response}")
    
    # JSON generation
    console.print("\n[bold]3. JSON Generation[/bold]")
    prompt = '''Extract the dimensions from this description and return as JSON:
"A flange with outer diameter 100mm, inner diameter 50mm, and thickness 10mm"

Return only: {"outer_diameter": <float>, "inner_diameter": <float>, "thickness": <float>}'''
    
    console.print(f"   Prompt: [dim]{prompt[:80]}...[/dim]")
    
    result = client.generate_json(prompt, max_tokens=100)
    console.print(f"   Result: {result}")
    
    # Structured output with Pydantic
    console.print("\n[bold]4. Structured Output (Pydantic)[/bold]")
    
    class FlangeParams(BaseModel):
        """Flange parameters extracted from description."""
        outer_diameter: float = Field(..., description="Outer diameter in mm")
        inner_diameter: float = Field(..., description="Inner diameter in mm")
        thickness: float = Field(..., description="Thickness in mm")
        num_holes: int = Field(default=0, description="Number of bolt holes")
    
    prompt = '''Extract flange parameters from this description:
"A steel flange with 120mm outer diameter, 60mm inner bore, 15mm thick, with 6 bolt holes"'''
    
    console.print(f"   Prompt: [dim]{prompt[:60]}...[/dim]")
    console.print(f"   Schema: FlangeParams")
    
    try:
        result = client.generate_structured(prompt, FlangeParams, max_tokens=150)
        console.print(f"   Result: {result}")
        console.print(f"   Type: {type(result).__name__}")
    except Exception as e:
        error_msg = str(e)
        if "Field required" in error_msg:
            console.print(f"   [yellow]Structured output error: Schema mismatch[/yellow]")
            console.print(f"   [dim]LLM returned JSON with different field names than expected[/dim]")
        else:
            console.print(f"   [yellow]Structured output error: {error_msg[:100]}...[/yellow]")
        
        # Try with explicit JSON instruction
        console.print("   Retrying with explicit JSON-only prompt...")
        json_prompt = '''Return ONLY a JSON object (no explanation) with these exact fields:
{"outer_diameter": 120.0, "inner_diameter": 60.0, "thickness": 15.0, "num_holes": 6}
Parse from: "A steel flange with 120mm outer diameter, 60mm inner bore, 15mm thick, with 6 bolt holes"'''
        try:
            raw_result = client.generate_json(json_prompt, max_tokens=100)
            console.print(f"   Raw JSON: {raw_result}")
        except Exception as e2:
            console.print(f"   [dim]JSON extraction also failed - LLM returned prose instead of JSON[/dim]")
    
    # Chat with system prompt
    console.print("\n[bold]5. Chat with System Prompt[/bold]")
    
    messages = [
        {"role": "system", "content": "You are a mechanical engineer. Be concise."},
        {"role": "user", "content": "What's the difference between a shaft and an axle?"}
    ]
    
    response = client.chat(messages, max_tokens=100)
    console.print(f"   Response: {response}")
    
    console.print("\n[green]✓ LLM client demo complete![/green]")


if __name__ == "__main__":
    main()

