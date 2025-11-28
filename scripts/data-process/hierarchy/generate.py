""" generate multi-label dataset based on config"""
import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random


def generate_hierarchical_dataset(source_dataset_path: str, output_path: str, config: Dict) -> None:
    """
    Generate hierarchical softmax dataset from original YOLO format dataset.
    
    Args:
        source_dataset_path: Path to source dataset (e.g., data2/which-bin)
        output_path: Path where hierarchical dataset will be created
        config: Configuration dictionary containing hierarchy mapping
    """
    source_path = Path(source_dataset_path)
    output_path = Path(output_path)
    
    # Load original data.yaml
    with open(source_path / "data.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Extract hierarchy from config
    hierarchy = config.get('hierarchy', {})
    class_mapping = {}
    
    # Create mapping from original classes to hierarchical structure
    for category, items in hierarchy.items():
        for item in items:
            if item in original_config['names']:
                original_idx = original_config['names'].index(item)
                class_mapping[original_idx] = category
    
    # Process each split
    for split in ['train', 'val', 'test']:
        source_split_path = source_path / split
        output_split_path = output_path / split
        
        if not source_split_path.exists():
            continue
            
        images_dir = source_split_path / 'images'
        labels_dir = source_split_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        # Process each image and its corresponding label
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            # Copy image
            shutil.copy2(img_file, output_split_path / 'images' / img_file.name)
            
            # Process corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            output_label_file = output_split_path / 'labels' / f"{img_file.stem}.txt"
            
            if label_file.exists():
                process_label_file(label_file, output_label_file, class_mapping, hierarchy)
    
    # Create new data.yaml for hierarchical dataset
    create_hierarchical_data_yaml(output_path, hierarchy, original_config)
    
    print(f"Hierarchical dataset generated at: {output_path}")


def process_label_file(source_label: Path, output_label: Path, class_mapping: Dict, hierarchy: Dict) -> None:
    """
    Process a single label file, converting original class indices to hierarchical format.
    
    Args:
        source_label: Source label file path
        output_label: Output label file path
        class_mapping: Mapping from original class indices to categories
        hierarchy: Hierarchical structure dictionary
    """
    new_labels = []
    category_names = list(hierarchy.keys())
    
    with open(source_label, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                original_class_idx = int(parts[0])
                bbox_coords = parts[1:5]
                
                # Map to hierarchical category
                if original_class_idx in class_mapping:
                    category = class_mapping[original_class_idx]
                    new_class_idx = category_names.index(category)
                    
                    new_label = f"{new_class_idx} " + " ".join(bbox_coords)
                    if len(parts) > 5:  # Include confidence if present
                        new_label += " " + " ".join(parts[5:])
                    
                    new_labels.append(new_label)
    
    # Write processed labels
    with open(output_label, 'w') as f:
        for label in new_labels:
            f.write(label + '\n')


def create_hierarchical_data_yaml(output_path: Path, hierarchy: Dict, original_config: Dict) -> None:
    """
    Create data.yaml file for hierarchical dataset.
    
    Args:
        output_path: Output dataset path
        hierarchy: Hierarchical structure dictionary
        original_config: Original dataset configuration
    """
    hierarchical_config = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(hierarchy),
        'names': list(hierarchy.keys())
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(hierarchical_config, f, default_flow_style=False)
    
    # Save hierarchy mapping for reference
    hierarchy_info = {
        'hierarchy': hierarchy,
        'original_classes': original_config.get('names', []),
        'mapping_info': 'Original classes mapped to hierarchical categories'
    }
    
    with open(output_path / 'hierarchy_mapping.yaml', 'w') as f:
        yaml.dump(hierarchy_info, f, default_flow_style=False)


def load_config(config_path: str = "../../data/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def example_usage():
    """Example of how to use the hierarchical dataset generator."""
    
    # Example hierarchy configuration for waste sorting
    hierarchy_config = {
        'hierarchy': {
            'RECYCLABLE': ['METAL', 'GLASS', 'PET_CONTAINER', 'HDPE_CONTAINER', 'PAPER', 'CARDBOARD'],
            'NON_RECYCLABLE': ['PLASTIC_WRAPPER', 'PLASTIC_BAG', 'PAPER_CUP', 'PAPER_BAG', 'USED_TISSUE', 'STYROFOAM', 'PLASTIC_CUP'],
            'SPECIAL_WASTE': ['TETRAPAK']
        }
    }
    
    # Generate hierarchical dataset
    generate_hierarchical_dataset(
        source_dataset_path='../../data2/which-bin',
        output_path='../../data2/which-bin-hierarchical',
        config=hierarchy_config
    )


if __name__ == "__main__":
    example_usage()