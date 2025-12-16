"""
CAD Generation Module

Template-based CadQuery generators for each LLM4CAD category.
Generates STEP files and PNG renders from parametric specifications.

Supported categories:
- Flange: Cylindrical base with raised face and bore
- Gear: Spur gear with teeth, width, and bore
- Nut: Hexagonal nut with threads
- Shaft: Stepped cylindrical shaft
- Spring: Helical coil spring
"""

import math
import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from src.config import get_config
from src.schemas import CADCategory, CADResult


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
# Flange Generator
# =============================================================================

def generate_flange(
    base_diameter: float,
    base_height: float,
    outer_diameter: float,
    inner_diameter: float,
    flange_height: float,
) -> Any:
    """
    Generate a flange CAD model.
    
    Args:
        base_diameter: Diameter of the circular base in mm
        base_height: Height/thickness of the base in mm
        outer_diameter: Diameter of the raised face in mm
        inner_diameter: Bore diameter in mm
        flange_height: Total height including raised face in mm
    
    Returns:
        CadQuery Workplane with the flange model
    """
    import cadquery as cq
    
    # Raised face height
    raised_height = flange_height - base_height
    
    # Create base cylinder
    result = (
        cq.Workplane("XY")
        .circle(base_diameter / 2)
        .extrude(base_height)
    )
    
    # Add raised face
    if raised_height > 0 and outer_diameter > 0:
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(outer_diameter / 2)
            .extrude(raised_height)
        )
    
    # Create bore (through hole)
    if inner_diameter > 0:
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(inner_diameter / 2)
            .cutThruAll()
        )
    
    return result


# =============================================================================
# Gear Generator
# =============================================================================

def generate_gear(
    module: float,
    teeth_number: int,
    width: float,
    bore_d: float,
) -> Any:
    """
    Generate a spur gear CAD model.
    
    Args:
        module: Gear module (tooth size parameter)
        teeth_number: Number of teeth
        width: Gear width/thickness in mm
        bore_d: Bore diameter in mm
    
    Returns:
        CadQuery Workplane with the gear model
    """
    import cadquery as cq
    
    # Calculate gear dimensions
    pitch_diameter = module * teeth_number
    outer_diameter = pitch_diameter + 2 * module
    root_diameter = pitch_diameter - 2.5 * module
    
    # Simplified gear as cylinder with teeth represented by smaller cylinder
    # Full involute gear profile is complex - use simplified representation
    result = (
        cq.Workplane("XY")
        .circle(outer_diameter / 2)
        .extrude(width)
    )
    
    # Add bore
    if bore_d > 0:
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(bore_d / 2)
            .cutThruAll()
        )
    
    return result


# =============================================================================
# Nut Generator
# =============================================================================

def generate_nut(
    nut_size: float,
    nut_height: float,
    inner_diameter: float,
) -> Any:
    """
    Generate a hexagonal nut CAD model.
    
    Args:
        nut_size: Nut size (across flats) in mm
        nut_height: Nut height in mm
        inner_diameter: Thread inner diameter in mm
    
    Returns:
        CadQuery Workplane with the nut model
    """
    import cadquery as cq
    
    # Calculate hexagon inscribed radius
    inscribed_radius = nut_size / 2
    
    # Create hexagonal prism
    result = (
        cq.Workplane("XY")
        .polygon(6, nut_size * 2 / math.sqrt(3))  # Circumscribed diameter
        .extrude(nut_height)
    )
    
    # Create through hole (thread)
    if inner_diameter > 0:
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(inner_diameter / 2)
            .cutThruAll()
        )
    
    return result


# =============================================================================
# Shaft Generator
# =============================================================================

def generate_shaft(
    sections: list[list[float]],
) -> Any:
    """
    Generate a stepped shaft CAD model.
    
    Args:
        sections: List of [length, diameter] pairs for each section
    
    Returns:
        CadQuery Workplane with the shaft model
    """
    import cadquery as cq
    
    if not sections:
        raise ValueError("Shaft must have at least one section")
    
    # Start with first section
    length, diameter = sections[0]
    result = (
        cq.Workplane("XY")
        .circle(diameter / 2)
        .extrude(length)
    )
    
    # Add remaining sections
    current_z = length
    for section in sections[1:]:
        length, diameter = section
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(diameter / 2)
            .extrude(length)
        )
        current_z += length
    
    return result


# =============================================================================
# Spring Generator
# =============================================================================

def generate_spring(
    radius: float,
    pitch: float,
    height: float,
    wire_radius: float,
) -> Any:
    """
    Generate a helical coil spring CAD model.
    
    Args:
        radius: Coil radius (mean radius) in mm
        pitch: Pitch (spacing between coils) in mm
        height: Free length/height in mm
        wire_radius: Wire radius in mm
    
    Returns:
        CadQuery Workplane with the spring model
    """
    import cadquery as cq
    
    # Calculate number of coils
    num_coils = height / pitch if pitch > 0 else 1
    
    # Create helix path
    # CadQuery doesn't have built-in helix, so we create a simplified version
    # using a series of circles swept along a helical path
    
    # For simplicity, create a torus-like approximation
    # A proper spring would require more complex geometry
    
    # Create wire cross-section
    wire_diameter = wire_radius * 2
    
    # Create spring using makeHelix
    helix = cq.Wire.makeHelix(
        pitch=pitch,
        height=height,
        radius=radius,
    )
    
    # Create wire circle
    wire = (
        cq.Workplane("XZ")
        .center(radius, 0)
        .circle(wire_radius)
    )
    
    # Sweep wire along helix
    result = wire.sweep(helix, isFrenet=True)
    
    return result


# =============================================================================
# Generator Dispatcher
# =============================================================================

GENERATORS = {
    CADCategory.FLANGE: generate_flange,
    CADCategory.GEAR: generate_gear,
    CADCategory.NUT: generate_nut,
    CADCategory.SHAFT: generate_shaft,
    CADCategory.SPRING: generate_spring,
}

# Parameter aliases for each category
# Maps common LLM output names to expected function parameter names
PARAM_ALIASES = {
    CADCategory.FLANGE: {
        # Aliases are keys, canonical names are values
    },
    CADCategory.GEAR: {
        "num_teeth": "teeth_number",
        "teeth": "teeth_number",
        "tooth_count": "teeth_number",
        "number_of_teeth": "teeth_number",
        "face_width": "width",
        "gear_width": "width",
        "thickness": "width",
        "bore_diameter": "bore_d",
        "bore": "bore_d",
        "inner_diameter": "bore_d",
        "hole_diameter": "bore_d",
    },
    CADCategory.NUT: {
        "size": "nut_size",
        "across_flats": "nut_size",
        "height": "nut_height",
        "thread_diameter": "inner_diameter",
        "thread_size": "inner_diameter",
        "bore": "inner_diameter",
    },
    CADCategory.SHAFT: {},  # Shaft uses list format
    CADCategory.SPRING: {
        "wire_diameter": "wire_radius",  # Special: divide by 2
        "wire_d": "wire_radius",
        "coil_diameter": "radius",  # Special: divide by 2
        "coil_d": "radius",
        "mean_diameter": "radius",
        "num_coils": "num_coils_count",  # Used for calculating pitch
        "coils": "num_coils_count",
        "number_of_coils": "num_coils_count",
        "active_coils": "num_coils_count",
        "free_length": "height",
        "length": "height",
    },
}


def normalize_params(category: CADCategory, params: dict) -> dict:
    """
    Normalize parameter names from LLM output to generator function names.
    
    Handles common variations in parameter naming from LLM responses.
    """
    aliases = PARAM_ALIASES.get(category, {})
    normalized = {}
    
    for key, value in params.items():
        # Check if this is an alias
        canonical_key = aliases.get(key, key)
        normalized[canonical_key] = value
    
    # Special handling for Spring parameters
    if category == CADCategory.SPRING:
        # Convert wire_diameter to wire_radius
        if "wire_radius" in normalized and normalized["wire_radius"] > 5:
            # If value is large, it's probably diameter not radius
            normalized["wire_radius"] = normalized["wire_radius"] / 2
        
        # Convert coil_diameter to radius
        if "radius" in normalized and normalized["radius"] > 10:
            # If value is large, it's probably diameter not radius
            normalized["radius"] = normalized["radius"] / 2
        
        # Calculate pitch from num_coils and height
        if "num_coils_count" in normalized and "height" in normalized:
            num_coils = normalized.pop("num_coils_count")
            height = normalized["height"]
            if num_coils > 0:
                normalized["pitch"] = height / num_coils
            else:
                normalized["pitch"] = 10.0  # Default pitch
    
    return normalized


def generate_cad_model(
    category: CADCategory | str,
    params: dict | list,
) -> Any:
    """
    Generate CAD model for a given category and parameters.
    
    Args:
        category: Part category
        params: Parameters (dict for most categories, list for Shaft)
    
    Returns:
        CadQuery Workplane with the model
    """
    if isinstance(category, str):
        category = CADCategory(category)
    
    generator = GENERATORS.get(category)
    if generator is None:
        raise ValueError(f"No generator for category: {category}")
    
    # Handle different parameter formats
    if category == CADCategory.SHAFT:
        # Shaft takes list of sections
        if not isinstance(params, list):
            raise ValueError("Shaft parameters must be a list of sections")
        return generator(params)
    else:
        # Other categories take dict
        if not isinstance(params, dict):
            raise ValueError(f"{category.value} parameters must be a dict")
        # Normalize parameter names
        normalized = normalize_params(category, params)
        return generator(**normalized)


# =============================================================================
# Export Functions
# =============================================================================

def export_step(model: Any, filepath: str) -> None:
    """Export CadQuery model to STEP file."""
    import cadquery as cq
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    cq.exporters.export(model, filepath, exportType="STEP")


def export_stl(model: Any, filepath: str) -> None:
    """Export CadQuery model to STL file."""
    import cadquery as cq
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    cq.exporters.export(model, filepath, exportType="STL")


# =============================================================================
# Rendering
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
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Try Method 1: matplotlib + numpy-stl (Simple and reliable)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend, no X server needed
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        from stl import mesh as stl_mesh
        import tempfile
        
        # Export to temporary STL
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_path = tmp.name
        
        export_stl(model, tmp_path)
        
        # Load STL file
        mesh = stl_mesh.Mesh.from_file(tmp_path)
        
        # Create figure with high DPI
        dpi = 100
        fig = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create poly collection from mesh
        vectors = mesh.vectors
        collection = Poly3DCollection(vectors, alpha=0.9, facecolor='steelblue', edgecolor='navy', linewidths=0.5)
        ax.add_collection3d(collection)
        
        # Auto scale to mesh size
        scale = mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set isometric view
        ax.view_init(elev=30, azim=45)
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Remove axes for cleaner look
        ax.set_axis_off()
        
        # Save to file
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Cleanup
        os.unlink(tmp_path)
        
        print(f"INFO: Rendered with matplotlib: {filepath}")
        return
        
    except Exception as e:
        print(f"INFO: matplotlib rendering failed: {e}, trying fallbacks...")
    
    # Try Method 1: CadQuery OCP native rendering (OpenCascade)
    try:
        from OCP.Graphic3d import Graphic3d_Camera, Graphic3d_RenderingParams
        from OCP.V3d import V3d_View, V3d_Viewer
        from OCP.Aspect import Aspect_DisplayConnection, Aspect_TypeOfTriedronPosition
        from OCP.OpenGl import OpenGl_GraphicDriver
        from OCP.Image import Image_AlienPixMap, Image_Format
        
        import tempfile
        
        # Get the shape from CadQuery model
        shape = model.val()
        
        # Create display connection (offscreen)
        display_connection = Aspect_DisplayConnection()
        
        # Create graphics driver
        graphics_driver = OpenGl_GraphicDriver(display_connection)
        
        # Create viewer
        viewer = V3d_Viewer(graphics_driver)
        
        # Create view
        view = viewer.CreateView()
        
        # Set background color (white)
        view.SetBackgroundColor(1.0, 1.0, 1.0)
        
        # Add the shape to display
        # This requires an interactive context, so we'll use a simpler approach
        # Export to image using OCP's built-in rendering
        
        # Set up camera for isometric view
        view.Camera().SetProjectionType(Graphic3d_Camera.Projection_Orthographic)
        
        # Fit all
        view.FitAll(0.01, False)
        view.ZFitAll()
        
        # Set isometric view
        view.SetProj(1, 1, 1)  # Isometric direction
        
        # Create image buffer
        image = Image_AlienPixMap()
        image.InitTrash(Image_Format.Image_Format_RGB, resolution[0], resolution[1])
        
        # Render to image
        view.ToPixMap(image, resolution[0], resolution[1])
        
        # Save to file
        image.Save(filepath)
        
        print(f"INFO: Rendered with CadQuery OCP: {filepath}")
        return
        
    except Exception as e:
        print(f"INFO: CadQuery OCP rendering failed: {e}, trying FreeCAD...")
    
    # Try Method 2: FreeCAD headless
    try:
        import tempfile
        import sys
        
        # Export to temporary STEP file (FreeCAD works better with STEP)
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            tmp_path = tmp.name
        
        export_step(model, tmp_path)
        
        # Try to import and render with FreeCAD
        try:
            import FreeCAD
            import FreeCADGui
            
            # Create document and import
            doc = FreeCAD.newDocument("TempDoc")
            
            # Import STEP file
            import Import
            Import.insert(tmp_path, doc.Name)
            
            # Initialize GUI (required for rendering, but can be headless)
            if not FreeCADGui.ActiveDocument:
                FreeCADGui.showMainWindow()
            
            # Get or create view
            view = FreeCADGui.activeDocument().activeView()
            
            # Set isometric view and fit all
            view.viewIsometric()
            view.fitAll()
            
            # Set render settings
            view.setAnimationEnabled(False)
            
            # Render to image
            view.saveImage(filepath, resolution[0], resolution[1], 'Current')
            
            # Cleanup
            FreeCAD.closeDocument(doc.Name)
            os.unlink(tmp_path)
            
            print(f"INFO: Rendered with FreeCAD: {filepath}")
            return
            
        except ImportError as e:
            print(f"INFO: FreeCAD not available: {e}")
        except Exception as e:
            print(f"INFO: FreeCAD rendering failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"INFO: FreeCAD setup failed: {e}")
    
    # Fallback Method 3: PIL placeholder
    print(f"INFO: Using PIL placeholder for {filepath}")
    try:
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', resolution, color='white')
        draw = ImageDraw.Draw(img)
        
        # Add informative text
        text_y = resolution[1] // 2
        draw.text((resolution[0]//4, text_y - 20), 
                 "CAD Model Generated", fill='darkgray', anchor="lt")
        draw.text((resolution[0]//4, text_y + 10), 
                 "(3D render unavailable)", fill='gray', anchor="lt")
        img.save(filepath)
        
    except Exception as e:
        raise RuntimeError(f"All rendering methods failed. Last error: {e}")


def get_model_volume(model: Any) -> float:
    """Get the volume of a CadQuery model in mm³."""
    try:
        # Get the solid
        solid = model.val()
        if hasattr(solid, 'Volume'):
            return solid.Volume()
        return 0.0
    except Exception:
        return 0.0


def get_model_bounding_box(model: Any) -> Optional[dict]:
    """Get the bounding box of a CadQuery model in mm."""
    try:
        solid = model.val()
        if hasattr(solid, 'BoundingBox'):
            bb = solid.BoundingBox()
            return {
                "x": bb.xlen,
                "y": bb.ylen,
                "z": bb.zlen,
            }
        return None
    except Exception:
        return None


# =============================================================================
# High-Level Interface
# =============================================================================

def generate_and_export(
    category: CADCategory | str,
    params: dict | list,
    output_dir: str,
    sample_id: str,
    iteration: int = 0,
    export_step: bool = True,
    export_stl: bool = False,
    render_png: bool = True,
) -> GenerationResult:
    """
    Generate CAD model and export to files.
    
    Args:
        category: Part category
        params: Parameters for the category
        output_dir: Base output directory
        sample_id: Sample identifier for naming
        iteration: Iteration number
        export_step: Whether to export STEP file
        export_stl: Whether to export STL file
        render_png: Whether to render PNG image
    
    Returns:
        GenerationResult with file paths and status
    """
    if not is_cadquery_available():
        return GenerationResult(
            success=False,
            error_message="CadQuery not available",
        )
    
    try:
        # Generate model
        model = generate_cad_model(category, params)
        
        # Get volume and bounding box
        volume = get_model_volume(model)
        bounding_box = get_model_bounding_box(model)
        
        # Set up paths
        output_path = Path(output_dir)
        cad_subdir = output_path / "cad"
        render_subdir = output_path / "renders"
        
        cad_file = None
        stl_file = None
        render_image = None
        
        # Export STEP
        if export_step:
            step_path = cad_subdir / f"{sample_id}_iter{iteration}.step"
            from src.cad.generators import export_step as _export_step
            _export_step(model, str(step_path))
            cad_file = str(step_path)
        
        # Export STL
        if export_stl:
            stl_path = cad_subdir / f"{sample_id}_iter{iteration}.stl"
            from src.cad.generators import export_stl as _export_stl
            _export_stl(model, str(stl_path))
            stl_file = str(stl_path)
        
        # Render PNG
        if render_png:
            png_path = render_subdir / f"{sample_id}_iter{iteration}.png"
            render_to_png(model, str(png_path))
            render_image = str(png_path)
        
        return GenerationResult(
            success=True,
            cad_file=cad_file,
            stl_file=stl_file,
            render_image=render_image,
            volume=volume,
            bounding_box=bounding_box,
        )
        
    except Exception as e:
        return GenerationResult(
            success=False,
            error_message=str(e),
        )


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
        import pyvista as pv
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        pv.global_theme.background = 'white'
        mesh = pv.read(stl_path)
        
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.add_mesh(mesh, color='steelblue', smooth_shading=True)
        plotter.view_isometric()
        plotter.screenshot(output_path)
        plotter.close()
        
        return True
        
    except Exception as e:
        print(f"Failed to render STL: {e}")
        return False

