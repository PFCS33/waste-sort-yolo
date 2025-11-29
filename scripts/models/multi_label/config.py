"""
Load hierarchy configuration info
"""

from ...data.utils import load_config
from pathlib import Path


class HierarchyConfig:
    """Hierarchy configuration manager."""

    def __init__(self, config_path):

        config = load_config(config_path)

        self.nc_material = config["nc_m"]  # e.g. 5
        self.nc_object = config["nc_o"]  # e.g. 14
        self.nc_total = self.nc_material + self.nc_object  # e.g. 19

        self.material_names = config["names_m"]
        self.object_names = config["names_o"]

        # original mapping: object_idx (0-13) → material_idx (0-4)
        self._raw_mapping = {int(k): int(v) for k, v in config["mapping"].items()}

        # transfomed mapping: class_id (5-18) → material_idx (0-4)
        self.object_to_material = {
            k + self.nc_material: v for k, v in self._raw_mapping.items()
        }

    def get_material_id(self, class_id):
        """
        Get material ID from class ID.l
        """
        if class_id < self.nc_material:
            return class_id
        return self.object_to_material.get(class_id, -1)

    def is_object_class(self, class_id: int) -> bool:
        """Check if class_id is an object (not material)."""
        return class_id >= self.nc_material

    def is_material_class(self, class_id: int) -> bool:
        """Check if class_id is a material."""
        return class_id < self.nc_material
