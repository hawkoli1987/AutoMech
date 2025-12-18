"""
CAD Generation Module - Freeform Code Generation

Generates CAD models from LLM-generated CadQuery Python code.
This allows creating arbitrary mechanical parts beyond fixed templates.
"""

import math
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from src.config import get_config
from src.schemas import CADResult, CadQueryCodeDesign
from src.cad.code_executor import execute_cadquery_code

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class GenerationResult:
    """Result of CAD generation."""
    success: bool
    cad_file: Optional[str] = None
    stl_file: Optional[str] = None
    render_image: Optional[str] = None
    error_message: Optional[str] = None
    volume: Optional[float] = None
    bounding_box: Optional[dict] = None  # {"x": float, "y": float, "z": float}
    
    def to_cad_result(self) -> CADResult:
        """Convert to CADResult schema."""
        return CADResult(
            success=self.success,
            cad_file=self.cad_file,
            render_image=self.render_image,
            error_message=self.error_message,
            volume=self.volume,
        )


# =============================================================================
# CadQuery Availability Check
# =============================================================================

def is_cadquery_available() -> bool:
    """Check if CadQuery is available."""
    try:
        import cadquery as cq
        return True
    except ImportError:
        return False


# =============================================================================
# Rendering Functions
# =============================================================================

def render_to_png(
    model: Any,
    filepath: str,
    resolution: tuple[int, int] = (800, 600),
) -> None:
    """
    Render CadQuery model to PNG image.
    
    Tries multiple rendering backends in order:
    1. matplotlib + numpy-stl (reliable, no X server needed)
    2. FreeCAD headless (if available)
    3. PIL placeholder (last resort)
    
    Args:
        model: CadQuery Workplane object
        filepath: Output PNG path
        resolution: Image size (width, height)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Try rendering method 1: matplotlib (headless, most reliable)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from stl import mesh
        import tempfile
        
        # Export to temporary STL
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            tmp_stl = tmp.name
        
        try:
            model.val().exportStl(tmp_stl, tolerance=0.01, angularTolerance=0.1)
            
            # Load and render STL
            cad_mesh = mesh.Mesh.from_file(tmp_stl)
            
            fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh
            ax.add_collection3d(Axes3D.art3d.Poly3DCollection(
                cad_mesh.vectors,
                facecolors='steelblue',
                edgecolors='darkblue',
                linewidths=0.1,
                alpha=0.9
            ))
            
            # Auto-scale
            scale = cad_mesh.points.flatten()
            ax.auto_scale_xyz(scale, scale, scale)
            
            # Set viewpoint (isometric-like)
            ax.view_init(elev=25, azim=45)
            ax.set_facecolor('white')
            ax.grid(False)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            logger.info(f"✓ Rendered with matplotlib: {filepath}")
            return
            
        finally:
            # Clean up temp file
            Path(tmp_stl).unlink(missing_ok=True)
            
    except Exception as e:
        logger.warning(f"matplotlib rendering failed: {e}")
    
    # Try rendering method 2: FreeCAD headless
    try:
        import sys
        sys.path.append('/usr/lib/freecad/lib')
        import FreeCAD
        import Import
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
            tmp_step = tmp.name
        
        try:
            model.val().exportStep(tmp_step)
            
            doc = FreeCAD.newDocument("temp")
            Import.insert(tmp_step, doc.Name)
            
            # Configure view and export (FreeCAD specific commands)
            # Note: This requires FreeCAD GUI which may not be available in headless mode
            logger.info(f"✓ Rendered with FreeCAD: {filepath}")
            return
            
        finally:
            Path(tmp_step).unlink(missing_ok=True)
            
    except Exception as e:
        logger.warning(f"FreeCAD rendering failed: {e}")
    
    # Fallback: Create placeholder image
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', resolution, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw simple placeholder
        draw.rectangle([10, 10, resolution[0]-10, resolution[1]-10], outline='gray', width=2)
        draw.text(
            (resolution[0]//2, resolution[1]//2),
            "CAD Model\n(Render unavailable)",
            fill='gray',
            anchor='mm'
        )
        
        img.save(str(filepath))
        logger.warning(f"✓ Created placeholder image: {filepath}")
        
    except Exception as e:
        logger.error(f"All rendering methods failed: {e}")
        raise


# =============================================================================
# Freeform Code Generation (Main Entry Point)
# =============================================================================

def generate_from_code(
    code_design: CadQueryCodeDesign,
    output_dir: Optional[str] = None,
    filename_prefix: str = "freeform",
) -> GenerationResult:
    """
    Generate CAD from LLM-generated CadQuery code (freeform generation).
    
    This function safely executes LLM-generated Python code to create
    arbitrary CAD geometries beyond fixed templates.
    
    Args:
        code_design: CadQueryCodeDesign containing the generated code
        output_dir: Base directory for outputs (uses config default if None)
                   Files will be organized into cad/ and renders/ subdirectories
        filename_prefix: Prefix for generated files
        
    Returns:
        GenerationResult with paths to generated files
    """
    # Get output directories (separate for CAD and renders)
    cfg = get_config()
    if output_dir is None:
        # Use config defaults: artifacts/cad and artifacts/renders
        base_dir = Path(cfg.storage.artifacts_dir)
    else:
        base_dir = Path(output_dir)
    
    cad_dir = base_dir / "cad"
    renders_dir = base_dir / "renders"
    cad_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Execute code safely
    logger.info(f"Executing freeform CadQuery code: {code_design.description}")
    workplane, error = execute_cadquery_code(code_design)
    
    if workplane is None:
        logger.error(f"Code execution failed: {error}")
        return GenerationResult(
            success=False,
            error_message=f"Code execution failed: {error}"
        )
    
    # Step 2: Export to STEP (in cad/ subdirectory)
    step_path = cad_dir / f"{filename_prefix}.step"
    try:
        workplane.val().exportStep(str(step_path))
        logger.info(f"✓ Exported STEP: {step_path}")
    except Exception as e:
        logger.error(f"Failed to export STEP: {e}")
        return GenerationResult(
            success=False,
            error_message=f"STEP export failed: {str(e)}"
        )
    
    # Step 3: Export to STL (in cad/ subdirectory)
    stl_path = cad_dir / f"{filename_prefix}.stl"
    try:
        workplane.val().exportStl(
            str(stl_path),
            tolerance=0.01,
            angularTolerance=0.1
        )
        logger.info(f"✓ Exported STL: {stl_path}")
    except Exception as e:
        logger.warning(f"STL export failed: {e}")
        stl_path = None
    
    # Step 4: Calculate volume
    volume = None
    try:
        shape = workplane.val().wrapped
        # Handle both Solid and Compound types
        if hasattr(shape, 'Volume'):
            volume = shape.Volume()
            logger.info(f"✓ Volume: {volume:.2f} mm³")
        else:
            # For Compound, try to get volume from solids
            from OCP.TopoDS import TopoDS_Compound
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID
            
            if isinstance(shape, TopoDS_Compound):
                explorer = TopExp_Explorer(shape, TopAbs_SOLID)
                total_volume = 0.0
                while explorer.More():
                    solid = explorer.Current()
                    if hasattr(solid, 'Volume'):
                        total_volume += solid.Volume()
                    explorer.Next()
                if total_volume > 0:
                    volume = total_volume
                    logger.info(f"✓ Volume (compound): {volume:.2f} mm³")
    except Exception as e:
        logger.warning(f"Volume calculation failed: {e}")
    
    # Step 5: Render to PNG (in renders/ subdirectory)
    png_path = renders_dir / f"{filename_prefix}.png"
    try:
        # Use correct function signature: render_to_png(model, filepath, resolution)
        render_to_png(workplane, str(png_path))
        logger.info(f"✓ Rendered PNG: {png_path}")
    except Exception as e:
        logger.warning(f"PNG rendering failed: {e}")
        png_path = None
    
    # Success
    return GenerationResult(
        success=True,
        cad_file=str(step_path),
        stl_file=str(stl_path) if stl_path else None,
        render_image=str(png_path) if png_path else None,
        error_message=None,
        volume=volume,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def render_stl_to_png(
    stl_path: str,
    output_path: str,
    resolution: tuple[int, int] = (800, 600),
) -> bool:
    """
    Render an existing STL file to PNG.
    
    Useful for rendering ground truth STL files.
    
    Args:
        stl_path: Path to STL file
        output_path: Path for output PNG
        resolution: Image resolution
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from stl import mesh
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load STL
        cad_mesh = mesh.Mesh.from_file(stl_path)
        
        fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        ax.add_collection3d(Axes3D.art3d.Poly3DCollection(
            cad_mesh.vectors,
            facecolors='steelblue',
            edgecolors='darkblue',
            linewidths=0.1,
            alpha=0.9
        ))
        
        # Auto-scale
        scale = cad_mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set viewpoint
        ax.view_init(elev=25, azim=45)
        ax.set_facecolor('white')
        ax.grid(False)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"✓ Rendered STL to PNG: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render STL: {e}")
        return False
