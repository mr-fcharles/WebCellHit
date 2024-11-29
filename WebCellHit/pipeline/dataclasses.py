from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

@dataclass
class PreprocessPaths:
    """Container for model and data file paths
    
    Attributes:
        classifier_path: Path to classifier model
        classifier_mapper_path: Path to classifier mapper
        imputer_path: Path to imputer model
        celligner_path: Path to Celligner model
        tcga_data_path: Path to TCGA data
        tcga_metadata_path: Path to TCGA metadata
        tcga_code_map_path: Path to TCGA code map
        tcga_project_ids_path: Path to TCGA project ids
        umap_path: Path to UMAP model
    """
    classifier_path: Union[Path, str] = None
    classifier_mapper_path: Union[Path, str] = None
    imputer_path: Union[Path, str] = None
    celligner_path: Union[Path, str] = None
    tcga_data_path: Union[Path, str] = None
    tcga_metadata_path: Union[Path, str] = None
    tcga_code_map_path: Union[Path, str] = None
    tcga_project_ids_path: Union[Path, str] = None
    umap_path: Optional[Union[Path, str]] = None

    def __post_init__(self):
        """Converts all string paths to Path objects and checks if paths exist."""
        # Check if paths exist
        paths_to_check = [
            path for path in [
                self.classifier_path,
                self.classifier_mapper_path,
                self.imputer_path,
                self.celligner_path,
                self.tcga_data_path,
                self.tcga_metadata_path,
                self.tcga_code_map_path,
                self.tcga_project_ids_path,
                self.umap_path
            ] if path is not None
        ]

        # Convert all string paths to Path objects
        present_paths = []
        for path in paths_to_check:
            if path is not None and isinstance(path, str):
                path = Path(path)
                present_paths.append(path)

        # Check if paths exist and list all non-existing paths
        non_existing_paths = [path for path in present_paths if not path.exists()]
        if non_existing_paths:
            raise FileNotFoundError(f"Paths {non_existing_paths} do not exist.")


@dataclass
class InferencePaths:
    """Dataclass containing all paths needed for inference. Accepts both Path and string objects, converting strings to Paths in the constructor. Checks if paths exist."""
    # Base paths
    cellhit_data: Union[Path, str] = None

    # Neighbors paths
    ccle_transcr_neighs: Union[Path, str] = None
    tcga_transcr_neighs: Union[Path, str] = None
    ccle_response_neighs: Union[Path, str] = None
    tcga_response_neighs: Union[Path, str] = None
    
    # Models paths
    pretrained_models_path: Union[Path, str] = None

    # Drug stats path
    drug_stats: Union[Path, str] = None

    # Drug metadata path
    drug_metadata: Union[Path, str] = None

    # Quantile computer path
    quantile_computer: Union[Path, str] = None

    # Metadata paths
    ccle_metadata: Union[Path, str] = None
    tcga_metadata: Union[Path, str] = None

    def __post_init__(self):
        """Converts all string paths to Path objects and checks if paths exist."""
        # Check if paths exist
        paths_to_check = [
            path for path in [
                self.cellhit_data,
                self.ccle_transcr_neighs,
                self.tcga_transcr_neighs,
                self.ccle_response_neighs,
                self.tcga_response_neighs,
                self.pretrained_models_path,
                self.drug_stats,
                self.drug_metadata,
                self.quantile_computer,
                self.ccle_metadata,
                self.tcga_metadata
            ] if path is not None
        ]

        # Convert all string paths to Path objects
        present_paths = []
        for path in paths_to_check:
            #if not None and string
            if path is not None and isinstance(path, str):
                path = Path(path)
                present_paths.append(path)
        
        # Check if paths exist and list all non-existing paths
        non_existing_paths = [path for path in present_paths if not path.exists()]
        if non_existing_paths:
            raise FileNotFoundError(f"Paths {non_existing_paths} do not exist.")
        
