"""
LLM4CAD Dataset Loader

Loads and parses the LLM4CAD dataset with its various category formats.
Provides unified access to samples via LLM4CADSample dataclass.

Dataset Structure:
- Flange, Gear, Nut: Organized in category folders
  - {Category}/dimension_text/{category}_XXXXX.json
  - {Category}/mesh/{category}_XXXXX.stl
  - {Category}/img/{category}_XXXXX.png
  - {Category}/{category}_description.csv

- Shaft, Spring: At root level
  - dimension_text/{Category}_XXXXX.json
  - mesh/{Category}_XXXXX.stl
  - img/{Category}_XXXXX.png
  - {category}_description.csv
"""

import csv
import json
import random
from pathlib import Path
from typing import Iterator, Optional

from src.schemas import LLM4CADSample, CADCategory
from src.config import get_config


# =============================================================================
# Category-specific configurations
# =============================================================================

CATEGORY_CONFIG = {
    CADCategory.FLANGE: {
        "folder": "Flange",
        "nested": True,  # Data is in category subfolder
        "id_prefix": "flange_",
        "csv_name": "flange_description.csv",
        "case_sensitive_id": False,  # IDs are lowercase
    },
    CADCategory.GEAR: {
        "folder": "Gear",
        "nested": True,
        "id_prefix": "Gear_",
        "csv_name": "gear_description.csv",
        "case_sensitive_id": True,  # IDs are capitalized
    },
    CADCategory.NUT: {
        "folder": "Nut",
        "nested": True,
        "id_prefix": "Nut_",
        "csv_name": "nut_description.csv",
        "case_sensitive_id": True,
    },
    CADCategory.SHAFT: {
        "folder": None,  # At root level
        "nested": False,
        "id_prefix": "Shaft_",
        "csv_name": "shaft_description.csv",
        "case_sensitive_id": True,
    },
    CADCategory.SPRING: {
        "folder": None,
        "nested": False,
        "id_prefix": "Spring_",
        "csv_name": "spring_description.csv",
        "case_sensitive_id": True,
    },
}


# =============================================================================
# Data Loader
# =============================================================================

class LLM4CADLoader:
    """
    Loader for the LLM4CAD dataset.
    
    Handles the varying folder structures and provides a unified interface
    to iterate over samples.
    """
    
    def __init__(
        self,
        data_path: Optional[str | Path] = None,
        categories: Optional[list[CADCategory | str]] = None,
        max_samples_per_category: Optional[int] = None,
        shuffle: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to LLM4CAD data directory. Uses config if not provided.
            categories: List of categories to load. None = all categories.
            max_samples_per_category: Limit samples per category. None = all.
            shuffle: Whether to shuffle samples.
            random_seed: Random seed for shuffling.
        """
        config = get_config()
        
        if data_path is None:
            data_path = config.data.llm4cad_path
        
        self.data_path = Path(data_path)
        
        # Parse categories
        if categories is None:
            self.categories = list(CADCategory)
        else:
            self.categories = [
                CADCategory(c) if isinstance(c, str) else c 
                for c in categories
            ]
        
        self.max_samples_per_category = max_samples_per_category
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"LLM4CAD data path not found: {self.data_path}")
    
    def _get_category_paths(self, category: CADCategory) -> dict[str, Path]:
        """Get paths for a category's data files."""
        cfg = CATEGORY_CONFIG[category]
        
        if cfg["nested"]:
            base = self.data_path / cfg["folder"]
        else:
            base = self.data_path
        
        # For nested categories, the folder name is used as prefix
        # For root-level categories, files are directly in dimension_text/, mesh/, img/
        if cfg["nested"]:
            return {
                "dimension_text": base / "dimension_text",
                "mesh": base / "mesh",
                "img": base / "img",
                "csv": base / cfg["csv_name"],
            }
        else:
            return {
                "dimension_text": self.data_path / "dimension_text",
                "mesh": self.data_path / "mesh",
                "img": self.data_path / "img",
                "csv": self.data_path / cfg["csv_name"],
            }
    
    def _load_descriptions(self, category: CADCategory) -> dict[str, str]:
        """
        Load text descriptions from CSV.
        
        Returns dict mapping filename (e.g., 'flange_00000.json') to description.
        """
        paths = self._get_category_paths(category)
        csv_path = paths["csv"]
        
        if not csv_path.exists():
            return {}
        
        descriptions = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                answer = row.get("answer", "").strip()
                if filename and answer:
                    descriptions[filename] = answer
        
        return descriptions
    
    def _load_params(self, json_path: Path) -> dict:
        """Load parameters from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _find_sample_ids(self, category: CADCategory) -> list[str]:
        """
        Find all sample IDs for a category.
        
        Returns list of sample IDs (without extension), e.g., ['flange_00000', 'flange_00001']
        """
        cfg = CATEGORY_CONFIG[category]
        paths = self._get_category_paths(category)
        
        dim_text_path = paths["dimension_text"]
        if not dim_text_path.exists():
            return []
        
        # Find all JSON files matching the pattern
        prefix = cfg["id_prefix"]
        sample_ids = []
        
        for json_file in dim_text_path.glob("*.json"):
            name = json_file.stem  # e.g., 'flange_00000'
            # Filter to only files matching this category's prefix
            if name.lower().startswith(prefix.lower()):
                sample_ids.append(name)
        
        return sorted(sample_ids)
    
    def _normalize_sample_id(self, sample_id: str, category: CADCategory) -> str:
        """Normalize sample ID casing based on category config."""
        cfg = CATEGORY_CONFIG[category]
        if not cfg["case_sensitive_id"]:
            return sample_id.lower()
        return sample_id
    
    def load_sample(self, sample_id: str, category: CADCategory) -> Optional[LLM4CADSample]:
        """
        Load a single sample by ID.
        
        Args:
            sample_id: Sample ID (e.g., 'flange_00000')
            category: Category enum
        
        Returns:
            LLM4CADSample or None if not found
        """
        cfg = CATEGORY_CONFIG[category]
        paths = self._get_category_paths(category)
        
        # Build file paths
        json_file = paths["dimension_text"] / f"{sample_id}.json"
        stl_file = paths["mesh"] / f"{sample_id}.stl"
        img_file = paths["img"] / f"{sample_id}.png"
        
        # Check JSON exists (required)
        if not json_file.exists():
            # Try alternate casing
            alt_id = sample_id.lower() if sample_id[0].isupper() else sample_id.capitalize()
            json_file = paths["dimension_text"] / f"{alt_id}.json"
            if not json_file.exists():
                return None
            sample_id = alt_id
            stl_file = paths["mesh"] / f"{sample_id}.stl"
            img_file = paths["img"] / f"{sample_id}.png"
        
        # Load parameters
        try:
            gt_param_spec = self._load_params(json_file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Failed to load params for {sample_id}: {e}")
            return None
        
        # Load descriptions
        descriptions = self._load_descriptions(category)
        json_filename = f"{sample_id}.json"
        text_desc = descriptions.get(json_filename, "")
        
        # If description not found, try alternate casing
        if not text_desc:
            alt_filename = json_filename.lower() if json_filename[0].isupper() else json_filename.capitalize()
            text_desc = descriptions.get(alt_filename, f"A {category.value.lower()} mechanical component.")
        
        return LLM4CADSample(
            sample_id=sample_id,
            category=category,
            text_desc=text_desc,
            gt_param_spec=gt_param_spec,
            stl_path=str(stl_file) if stl_file.exists() else "",
        )
    
    def __iter__(self) -> Iterator[LLM4CADSample]:
        """Iterate over all samples."""
        all_samples = []
        
        for category in self.categories:
            sample_ids = self._find_sample_ids(category)
            
            # Apply limit
            if self.max_samples_per_category is not None:
                sample_ids = sample_ids[:self.max_samples_per_category]
            
            for sample_id in sample_ids:
                sample = self.load_sample(sample_id, category)
                if sample is not None:
                    all_samples.append(sample)
        
        # Shuffle if requested
        if self.shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(all_samples)
        
        yield from all_samples
    
    def __len__(self) -> int:
        """Get total number of samples (without loading them all)."""
        total = 0
        for category in self.categories:
            sample_ids = self._find_sample_ids(category)
            if self.max_samples_per_category is not None:
                total += min(len(sample_ids), self.max_samples_per_category)
            else:
                total += len(sample_ids)
        return total
    
    def get_category_counts(self) -> dict[str, int]:
        """Get sample counts per category."""
        counts = {}
        for category in self.categories:
            sample_ids = self._find_sample_ids(category)
            if self.max_samples_per_category is not None:
                counts[category.value] = min(len(sample_ids), self.max_samples_per_category)
            else:
                counts[category.value] = len(sample_ids)
        return counts


# =============================================================================
# Convenience Functions
# =============================================================================

def load_llm4cad_samples(
    data_path: Optional[str | Path] = None,
    categories: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
) -> list[LLM4CADSample]:
    """
    Load LLM4CAD samples as a list.
    
    Args:
        data_path: Path to LLM4CAD data directory
        categories: List of category names (e.g., ["Flange", "Gear"])
        max_samples: Maximum total samples to load
        shuffle: Whether to shuffle samples
    
    Returns:
        List of LLM4CADSample objects
    """
    loader = LLM4CADLoader(
        data_path=data_path,
        categories=categories,
        shuffle=shuffle,
    )
    
    samples = list(loader)
    
    if max_samples is not None:
        samples = samples[:max_samples]
    
    return samples


def get_sample_by_id(sample_id: str, data_path: Optional[str | Path] = None) -> Optional[LLM4CADSample]:
    """
    Load a single sample by its ID.
    
    Args:
        sample_id: Full sample ID (e.g., 'flange_00001')
        data_path: Path to LLM4CAD data directory
    
    Returns:
        LLM4CADSample or None if not found
    """
    # Infer category from ID prefix
    sample_id_lower = sample_id.lower()
    
    for category in CADCategory:
        prefix = CATEGORY_CONFIG[category]["id_prefix"].lower()
        if sample_id_lower.startswith(prefix.rstrip("_")):
            loader = LLM4CADLoader(data_path=data_path, categories=[category])
            return loader.load_sample(sample_id, category)
    
    return None

