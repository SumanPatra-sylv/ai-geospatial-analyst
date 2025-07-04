#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Corrected Version)
Generates correct, executable spatial analysis workflows from parsed queries.
"""

from typing import List, Dict, Set
from pprint import pprint
import re

# This try/except block is preserved from your code.
try:
    from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
except ImportError:
    # Fallback for standalone testing - using correct attribute names
    from enum import Enum
    
    class SpatialRelationship(Enum):
        WITHIN = "within"
        NOT_WITHIN = "not within"
        NEAR = "near"
        FAR_FROM = "far from"
    
    class SpatialConstraint:
        def __init__(self, feature_type: str, relationship: SpatialRelationship, distance_meters: int = None):
            self.feature_type = feature_type
            self.relationship = relationship
            self.distance_meters = distance_meters
    
    class ParsedQuery:
        def __init__(self, target: str, location: str, constraints: List[SpatialConstraint]):
            self.target = target
            self.location = location
            self.constraints = constraints


class WorkflowGenerator:
    """
    Generates a logical sequence of spatial operations from a ParsedQuery.
    This class is stateless; its methods do not depend on or modify instance state.
    """
    # All of your static methods are preserved and correct.
    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name); sanitized = re.sub(r'_+', '_', sanitized); return sanitized.strip('_').lower()

    @staticmethod
    def _add_step(workflow: List[Dict], operation: str, **kwargs):
        step = {'step': len(workflow) + 1, 'operation': operation, **kwargs}; workflow.append(step)

    @staticmethod
    def _get_or_create_feature_layer(workflow: List[Dict], loaded_layers: Set[str], feature_type: str) -> str:
        safe_feature_type = WorkflowGenerator._sanitize_name(feature_type); output_layer = f"layer_{safe_feature_type}"
        if output_layer not in loaded_layers:
            WorkflowGenerator._add_step(workflow, 'filter_by_category', input_layer='base_landuse', category=feature_type, output_layer=output_layer)
            loaded_layers.add(output_layer)
        return output_layer

    @staticmethod
    def _get_or_create_buffered_layer(workflow: List[Dict], loaded_layers: Set[str], input_layer: str, distance: int) -> str:
        output_layer = f"{input_layer}_buffer_{distance}m"
        if output_layer not in loaded_layers:
            WorkflowGenerator._add_step(workflow, 'buffer', input_layer=input_layer, distance=distance, output_layer=output_layer)
            loaded_layers.add(output_layer)
        return output_layer

    # --- FIX: The method signature is reverted to correctly expect the ParsedQuery OBJECT ---
    # The logic for building the object from a dict has been removed, as that logic now correctly
    # resides in the tasks.py file, where the QueryParser is called.
    def generate_workflow(self, parsed_query: ParsedQuery) -> List[Dict]:
        """
        Generates a complete spatial analysis workflow from a parsed query object.
        """
        # --- The rest of your existing, correct logic is preserved below ---
        
        if not isinstance(parsed_query, ParsedQuery):
             raise TypeError(f"generate_workflow expects a ParsedQuery object, but received {type(parsed_query)}")

        workflow: List[Dict] = []
        loaded_layers: Set[str] = set()

        self._add_step(workflow, 'load_osm_data', location=parsed_query.location, output_layer='base_landuse')
        loaded_layers.add('base_landuse')

        candidate_layer = self._get_or_create_feature_layer(workflow, loaded_layers, parsed_query.target)
        
        positive_constraints = [c for c in parsed_query.constraints if c.relationship in [SpatialRelationship.WITHIN, SpatialRelationship.NEAR]]
        negative_constraints = [c for c in parsed_query.constraints if c.relationship in [SpatialRelationship.NOT_WITHIN, SpatialRelationship.FAR_FROM]]

        for i, constraint in enumerate(positive_constraints):
            constraint_feature_layer = self._get_or_create_feature_layer(workflow, loaded_layers, constraint.feature_type)
            if constraint.distance_meters:
                op_layer = self._get_or_create_buffered_layer(workflow, loaded_layers, constraint_feature_layer, constraint.distance_meters)
            else:
                op_layer = constraint_feature_layer
            safe_constraint_type = self._sanitize_name(constraint.feature_type); constraint_desc = safe_constraint_type
            if constraint.distance_meters: constraint_desc += f"_buffer_{constraint.distance_meters}m"
            new_candidate_layer = f"candidates_intersect_{constraint_desc}_{i+1}"
            self._add_step(workflow, 'intersect', input_layers=[candidate_layer, op_layer], output_layer=new_candidate_layer)
            candidate_layer = new_candidate_layer

        for i, constraint in enumerate(negative_constraints):
            constraint_feature_layer = self._get_or_create_feature_layer(workflow, loaded_layers, constraint.feature_type)
            if constraint.distance_meters:
                op_layer = self._get_or_create_buffered_layer(workflow, loaded_layers, constraint_feature_layer, constraint.distance_meters)
            else:
                op_layer = constraint_feature_layer
            safe_constraint_type = self._sanitize_name(constraint.feature_type); constraint_desc = safe_constraint_type
            if constraint.distance_meters: constraint_desc += f"_buffer_{constraint.distance_meters}m"
            new_candidate_layer = f"candidates_difference_{constraint_desc}_{i+1}"
            self._add_step(workflow, 'difference', input_layers=[candidate_layer, op_layer], output_layer=new_candidate_layer)
            candidate_layer = new_candidate_layer

        target_layer_name = f"layer_{self._sanitize_name(parsed_query.target)}"
        if candidate_layer != target_layer_name:
            self._add_step(workflow, 'rename_layer', input_layer=candidate_layer, output_layer='final_result')

        return workflow


# Your existing test block is preserved and will now work correctly again.
if __name__ == '__main__':
    sample_query = ParsedQuery(
        target='park',
        location='New York, NY',
        constraints=[
            SpatialConstraint(
                feature_type='residential',
                relationship=SpatialRelationship.WITHIN,
                distance_meters=500
            ),
            SpatialConstraint(
                feature_type='commercial',
                relationship=SpatialRelationship.NOT_WITHIN,
                distance_meters=200
            )
        ]
    )

    complex_query = ParsedQuery(
        target='restaurant',
        location='San Francisco, CA',
        constraints=[
            SpatialConstraint(
                feature_type='residential',
                relationship=SpatialRelationship.NEAR,
                distance_meters=300
            ),
            SpatialConstraint(
                feature_type='school',
                relationship=SpatialRelationship.FAR_FROM,
                distance_meters=150
            ),
            SpatialConstraint(
                feature_type='park',  
                relationship=SpatialRelationship.WITHIN,
                distance_meters=None
            )
        ]
    )

    generator = WorkflowGenerator()
    
    print("✅ Correctly Generated Workflow Plan (Simple Query):")
    print("=" * 60)
    workflow1 = generator.generate_workflow(sample_query)
    pprint(workflow1, width=100)
    print(f"\nTotal steps: {len(workflow1)}")
    
    print("\n" + "=" * 60)
    print("✅ Complex Query Workflow Plan:")
    print("=" * 60)
    workflow2 = generator.generate_workflow(complex_query)
    pprint(workflow2, width=100)
    print(f"\nTotal steps: {len(workflow2)}")
    
    print("\n" + "=" * 60)
    print("Workflow Summary for Simple Query:")
    for step in workflow1:
        if step['operation'] == 'load_osm_data':
            print(f"Step {step['step']}: Load OSM data for {step['location']}")
        elif step['operation'] == 'filter_by_category':
            print(f"Step {step['step']}: Extract {step['category']} features")
        elif step['operation'] == 'buffer':
            print(f"Step {step['step']}: Create {step['distance']}m buffer around {step['input_layer']}")
        elif step['operation'] == 'intersect':
            print(f"Step {step['step']}: Find intersection between {' and '.join(step['input_layers'])}")
        elif step['operation'] == 'difference':
            print(f"Step {step['step']}: Remove overlap between {' and '.join(step['input_layers'])}")
        elif step['operation'] == 'rename_layer':
            print(f"Step {step['step']}: Set final result layer")