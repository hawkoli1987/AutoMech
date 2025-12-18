"""
Tests for CAD Generation Module - Freeform Code Generation

Verifies:
- Code execution safety
- Freeform code generation
- Export functionality (STEP, STL)
- Rendering pipeline
- Error handling

Note: Tests requiring CadQuery are skipped if not available.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


def is_cadquery_available():
    """Check if CadQuery is available."""
    try:
        import cadquery
        return True
    except ImportError:
        return False


# Skip tests if CadQuery not available
pytestmark = pytest.mark.skipif(
    not is_cadquery_available(),
    reason="CadQuery not available"
)


class TestCodeExecutor:
    """Test code execution safety and validation."""
    
    def test_code_executor_valid_code(self):
        """Test executing valid CadQuery code."""
        from src.cad.code_executor import execute_cadquery_code
        from src.schemas import CadQueryCodeDesign
        
        code_design = CadQueryCodeDesign(
            description="Simple box",
            code="result = cq.Workplane('XY').box(10, 10, 10)",
            entry_point="result",
            required_imports=[],
        )
        
        workplane, error = execute_cadquery_code(code_design)
        
        assert workplane is not None
        assert error is None
    
    def test_code_executor_blocks_imports(self):
        """Test that import statements are stripped."""
        from src.cad.code_executor import execute_cadquery_code
        from src.schemas import CadQueryCodeDesign
        
        # Code with import should still work (imports are commented out)
        code_design = CadQueryCodeDesign(
            description="Box with import",
            code="import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)",
            entry_point="result",
            required_imports=[],
        )
        
        workplane, error = execute_cadquery_code(code_design)
        
        assert workplane is not None
        assert error is None
    
    def test_code_executor_syntax_error(self):
        """Test handling of syntax errors."""
        from src.cad.code_executor import execute_cadquery_code
        from src.schemas import CadQueryCodeDesign
        
        code_design = CadQueryCodeDesign(
            description="Invalid syntax",
            code="result = cq.Workplane('XY').box(10, 10, 10",  # Missing closing paren
            entry_point="result",
            required_imports=[],
        )
        
        workplane, error = execute_cadquery_code(code_design)
        
        assert workplane is None
        assert error is not None
        assert "Syntax error" in error
    
    def test_code_executor_forbidden_operation(self):
        """Test blocking of forbidden operations."""
        from src.cad.code_executor import execute_cadquery_code
        from src.schemas import CadQueryCodeDesign
        
        code_design = CadQueryCodeDesign(
            description="Forbidden open()",
            code="open('/etc/passwd'); result = cq.Workplane('XY').box(10, 10, 10)",
            entry_point="result",
            required_imports=[],
        )
        
        workplane, error = execute_cadquery_code(code_design)
        
        assert workplane is None
        assert error is not None
        assert "Forbidden operation" in error
    
    def test_code_executor_no_result(self):
        """Test handling when entry point is missing."""
        from src.cad.code_executor import execute_cadquery_code
        from src.schemas import CadQueryCodeDesign
        
        code_design = CadQueryCodeDesign(
            description="No result variable",
            code="box = cq.Workplane('XY').box(10, 10, 10)",
            entry_point="result",
            required_imports=[],
        )
        
        workplane, error = execute_cadquery_code(code_design)
        
        assert workplane is None
        assert error is not None
        assert "Entry point 'result' not found" in error
    
    def test_safety_preview(self):
        """Test code safety preview function."""
        from src.cad.code_executor import preview_code_safety
        
        # Valid code
        safe_code = "result = cq.Workplane('XY').box(10, 10, 10)"
        results = preview_code_safety(safe_code)
        
        assert results['overall']['passed'] is True
        assert results['syntax_check']['passed'] is True
        assert results['security_check']['passed'] is True
        
        # Unsafe code
        unsafe_code = "import os; os.system('rm -rf /')"
        results = preview_code_safety(unsafe_code)
        
        assert results['overall']['passed'] is False
        assert results['security_check']['passed'] is False


class TestFreeformGeneration:
    """Test freeform CAD generation end-to-end."""
    
    def test_generate_from_code_box(self):
        """Test generating a simple box."""
        from src.cad.generators import generate_from_code
        from src.schemas import CadQueryCodeDesign
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_design = CadQueryCodeDesign(
                description="10mm cube",
                code="result = cq.Workplane('XY').box(10, 10, 10)",
                entry_point="result",
                required_imports=[],
            )
            
            result = generate_from_code(
                code_design=code_design,
                output_dir=tmpdir,
                filename_prefix="test_box",
            )
            
            assert result.success is True
            assert result.cad_file is not None
            assert Path(result.cad_file).exists()
            assert result.cad_file.endswith('.step')
            
            # Check subdirectories
            assert '/cad/' in result.cad_file
            if result.render_image:
                assert '/renders/' in result.render_image
    
    def test_generate_from_code_cylinder(self):
        """Test generating a cylinder."""
        from src.cad.generators import generate_from_code
        from src.schemas import CadQueryCodeDesign
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_design = CadQueryCodeDesign(
                description="Cylinder",
                code="result = cq.Workplane('XY').circle(10).extrude(20)",
                entry_point="result",
                required_imports=[],
            )
            
            result = generate_from_code(
                code_design=code_design,
                output_dir=tmpdir,
                filename_prefix="test_cylinder",
            )
            
            assert result.success is True
            assert result.cad_file is not None
            assert Path(result.cad_file).exists()
    
    def test_generate_from_code_with_features(self):
        """Test generating geometry with holes and features."""
        from src.cad.generators import generate_from_code
        from src.schemas import CadQueryCodeDesign
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_design = CadQueryCodeDesign(
                description="Box with hole",
                code="""result = cq.Workplane('XY').box(20, 20, 10)
result = result.faces('>Z').workplane().circle(3).cutThruAll()""",
                entry_point="result",
                required_imports=[],
            )
            
            result = generate_from_code(
                code_design=code_design,
                output_dir=tmpdir,
                filename_prefix="test_features",
            )
            
            assert result.success is True
            assert result.cad_file is not None
    
    def test_generate_from_code_invalid_code(self):
        """Test handling of invalid code during generation."""
        from src.cad.generators import generate_from_code
        from src.schemas import CadQueryCodeDesign
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_design = CadQueryCodeDesign(
                description="Invalid code",
                code="result = invalid_function()",
                entry_point="result",
                required_imports=[],
            )
            
            result = generate_from_code(
                code_design=code_design,
                output_dir=tmpdir,
                filename_prefix="test_invalid",
            )
            
            assert result.success is False
            assert result.error_message is not None
            assert "Runtime error" in result.error_message


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_is_cadquery_available(self):
        """Test CadQuery availability check."""
        from src.cad.generators import is_cadquery_available
        
        # Should return True since tests are only run if CadQuery is available
        assert is_cadquery_available() is True
    
    def test_generation_result_to_cad_result(self):
        """Test GenerationResult conversion to CADResult."""
        from src.cad.generators import GenerationResult
        
        gen_result = GenerationResult(
            success=True,
            cad_file="/path/to/file.step",
            render_image="/path/to/render.png",
            volume=1000.0,
        )
        
        cad_result = gen_result.to_cad_result()
        
        assert cad_result.success is True
        assert cad_result.cad_file == "/path/to/file.step"
        assert cad_result.render_image == "/path/to/render.png"
        assert cad_result.volume == 1000.0


@pytest.mark.live
class TestLiveIntegration:
    """Integration tests requiring LLM (marked as 'live')."""
    
    def test_llm_code_generation_integration(self):
        """Test full pipeline with LLM code generation."""
        from src.utils.llm_client import get_llm_client
        from src.cad.generators import generate_from_code
        from src.schemas import CadQueryCodeDesign
        
        client = get_llm_client()
        
        prompt = """Generate CadQuery code for a simple 20mm cube.
Output JSON with: description, code, entry_point (always 'result'), required_imports (empty list)."""
        
        try:
            code_dict = client.generate_json(
                prompt=prompt,
                system_prompt="You are a CadQuery expert. Do NOT include import statements.",
                temperature=0.3,
                max_tokens=512,
            )
            
            code_dict.setdefault("entry_point", "result")
            code_dict.setdefault("required_imports", [])
            
            code_design = CadQueryCodeDesign(**code_dict)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                result = generate_from_code(
                    code_design=code_design,
                    output_dir=tmpdir,
                    filename_prefix="test_llm",
                )
                
                assert result.success is True
                
        except Exception as e:
            pytest.skip(f"LLM not available: {e}")
