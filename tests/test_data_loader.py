"""
Tests for Step 1.2: LLM4CAD Data Loader

Verifies:
- Data loader can find and parse samples from all categories
- CSV descriptions are correctly loaded
- JSON parameters are correctly parsed
- Sample iteration works correctly
"""

import pytest
from pathlib import Path


# Skip all tests if data not available
pytestmark = pytest.mark.skipif(
    not Path("data/LLM4CAD").exists(),
    reason="LLM4CAD data not available"
)


class TestLLM4CADLoader:
    """Test the LLM4CAD data loader."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance."""
        from src.data_loop.loader import LLM4CADLoader
        return LLM4CADLoader(data_path="data/LLM4CAD")
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader.data_path.exists()
        assert len(loader.categories) == 5  # All categories
    
    def test_find_flange_samples(self, loader):
        """Test finding Flange sample IDs."""
        from src.schemas import CADCategory
        sample_ids = loader._find_sample_ids(CADCategory.FLANGE)
        assert len(sample_ids) > 0
        assert any("flange" in sid.lower() for sid in sample_ids)
    
    def test_find_gear_samples(self, loader):
        """Test finding Gear sample IDs."""
        from src.schemas import CADCategory
        sample_ids = loader._find_sample_ids(CADCategory.GEAR)
        assert len(sample_ids) > 0
        assert any("gear" in sid.lower() for sid in sample_ids)
    
    def test_find_nut_samples(self, loader):
        """Test finding Nut sample IDs."""
        from src.schemas import CADCategory
        sample_ids = loader._find_sample_ids(CADCategory.NUT)
        assert len(sample_ids) > 0
    
    def test_find_shaft_samples(self, loader):
        """Test finding Shaft sample IDs."""
        from src.schemas import CADCategory
        sample_ids = loader._find_sample_ids(CADCategory.SHAFT)
        assert len(sample_ids) > 0
    
    def test_find_spring_samples(self, loader):
        """Test finding Spring sample IDs."""
        from src.schemas import CADCategory
        sample_ids = loader._find_sample_ids(CADCategory.SPRING)
        assert len(sample_ids) > 0
    
    def test_load_flange_sample(self, loader):
        """Test loading a single Flange sample."""
        from src.schemas import CADCategory
        
        sample = loader.load_sample("flange_00000", CADCategory.FLANGE)
        assert sample is not None
        assert sample.sample_id == "flange_00000"
        assert sample.category == CADCategory.FLANGE
        assert "flange" in sample.text_desc.lower() or "diameter" in sample.text_desc.lower()
        assert "base_diameter" in sample.gt_param_spec
        assert sample.gt_param_spec["base_diameter"] == 124
    
    def test_load_gear_sample(self, loader):
        """Test loading a single Gear sample."""
        from src.schemas import CADCategory
        
        sample = loader.load_sample("Gear_00000", CADCategory.GEAR)
        assert sample is not None
        assert sample.category == CADCategory.GEAR
        assert "module" in sample.gt_param_spec
        assert "teeth_number" in sample.gt_param_spec
    
    def test_load_spring_sample(self, loader):
        """Test loading a single Spring sample."""
        from src.schemas import CADCategory
        
        sample = loader.load_sample("Spring_00000", CADCategory.SPRING)
        assert sample is not None
        assert sample.category == CADCategory.SPRING
        assert "radius" in sample.gt_param_spec or "pitch" in sample.gt_param_spec
    
    def test_load_shaft_sample(self, loader):
        """Test loading a single Shaft sample (has array format)."""
        from src.schemas import CADCategory
        
        sample = loader.load_sample("Shaft_00000", CADCategory.SHAFT)
        assert sample is not None
        assert sample.category == CADCategory.SHAFT
        # Shaft uses array format: [[diameter, length], ...] for stepped shafts
        assert isinstance(sample.gt_param_spec, list)
        # Each element is [length, diameter] pair
        assert len(sample.gt_param_spec) >= 1
        assert len(sample.gt_param_spec[0]) == 2
    
    def test_sample_to_graph_state(self, loader):
        """Test converting sample to GraphState."""
        from src.schemas import CADCategory
        
        sample = loader.load_sample("flange_00000", CADCategory.FLANGE)
        assert sample is not None
        
        state = sample.to_graph_state(run_id="test_run_001")
        assert state.text_desc == sample.text_desc
        assert state.gt_param_spec == sample.gt_param_spec
        assert state.metadata.sample_id == "flange_00000"
        assert state.metadata.category == CADCategory.FLANGE
        assert state.metadata.run_id == "test_run_001"
        assert state.iteration == 0
        assert state.done is False
    
    def test_loader_iteration(self, loader):
        """Test iterating over samples."""
        samples = list(loader)
        assert len(samples) > 0
        
        # Check at least some samples from each category
        categories_found = set(s.category for s in samples)
        assert len(categories_found) >= 3  # At least 3 categories
    
    def test_loader_length(self, loader):
        """Test getting loader length."""
        length = len(loader)
        assert length > 0
        assert length >= 1000  # Should have at least 1000 samples total
    
    def test_category_counts(self, loader):
        """Test getting counts per category."""
        counts = loader.get_category_counts()
        assert len(counts) == 5
        for category, count in counts.items():
            assert count > 0


class TestLoaderWithLimits:
    """Test loader with sample limits."""
    
    def test_max_samples_per_category(self):
        """Test limiting samples per category."""
        from src.data_loop.loader import LLM4CADLoader
        
        loader = LLM4CADLoader(
            data_path="data/LLM4CAD",
            max_samples_per_category=10
        )
        
        counts = loader.get_category_counts()
        for category, count in counts.items():
            assert count <= 10
    
    def test_single_category_loading(self):
        """Test loading only one category."""
        from src.data_loop.loader import LLM4CADLoader
        from src.schemas import CADCategory
        
        loader = LLM4CADLoader(
            data_path="data/LLM4CAD",
            categories=[CADCategory.FLANGE],
            max_samples_per_category=5
        )
        
        samples = list(loader)
        assert len(samples) == 5
        assert all(s.category == CADCategory.FLANGE for s in samples)
    
    def test_shuffle_samples(self):
        """Test shuffling samples."""
        from src.data_loop.loader import LLM4CADLoader
        
        loader1 = LLM4CADLoader(
            data_path="data/LLM4CAD",
            max_samples_per_category=10,
            shuffle=True,
            random_seed=42
        )
        
        loader2 = LLM4CADLoader(
            data_path="data/LLM4CAD",
            max_samples_per_category=10,
            shuffle=True,
            random_seed=42
        )
        
        samples1 = list(loader1)
        samples2 = list(loader2)
        
        # Same seed should give same order
        assert [s.sample_id for s in samples1] == [s.sample_id for s in samples2]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_llm4cad_samples(self):
        """Test load_llm4cad_samples function."""
        from src.data_loop.loader import load_llm4cad_samples
        
        samples = load_llm4cad_samples(
            data_path="data/LLM4CAD",
            max_samples=20
        )
        
        assert len(samples) == 20
    
    def test_get_sample_by_id(self):
        """Test get_sample_by_id function."""
        from src.data_loop.loader import get_sample_by_id
        
        sample = get_sample_by_id("flange_00001", data_path="data/LLM4CAD")
        assert sample is not None
        assert sample.sample_id == "flange_00001"
    
    def test_get_sample_by_id_gear(self):
        """Test get_sample_by_id for Gear category."""
        from src.data_loop.loader import get_sample_by_id
        
        sample = get_sample_by_id("Gear_00001", data_path="data/LLM4CAD")
        assert sample is not None
        assert "Gear" in sample.sample_id


class TestDataIntegrity:
    """Test data integrity checks."""
    
    def test_flange_params_valid(self):
        """Test Flange parameters have expected fields."""
        from src.data_loop.loader import LLM4CADLoader
        from src.schemas import CADCategory
        
        loader = LLM4CADLoader(data_path="data/LLM4CAD", categories=[CADCategory.FLANGE])
        sample = loader.load_sample("flange_00000", CADCategory.FLANGE)
        
        expected_fields = ["base_diameter", "base_height", "outer_diameter", "inner_diameter", "flange_height"]
        for field in expected_fields:
            assert field in sample.gt_param_spec, f"Missing field: {field}"
    
    def test_gear_params_valid(self):
        """Test Gear parameters have expected fields."""
        from src.data_loop.loader import LLM4CADLoader
        from src.schemas import CADCategory
        
        loader = LLM4CADLoader(data_path="data/LLM4CAD", categories=[CADCategory.GEAR])
        sample = loader.load_sample("Gear_00000", CADCategory.GEAR)
        
        expected_fields = ["module", "teeth_number", "width", "bore_d"]
        for field in expected_fields:
            assert field in sample.gt_param_spec, f"Missing field: {field}"
    
    def test_spring_params_valid(self):
        """Test Spring parameters have expected fields."""
        from src.data_loop.loader import LLM4CADLoader
        from src.schemas import CADCategory
        
        loader = LLM4CADLoader(data_path="data/LLM4CAD", categories=[CADCategory.SPRING])
        sample = loader.load_sample("Spring_00000", CADCategory.SPRING)
        
        expected_fields = ["radius", "pitch", "height", "wire_radius"]
        for field in expected_fields:
            assert field in sample.gt_param_spec, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

