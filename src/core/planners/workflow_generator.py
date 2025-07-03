#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Corrected Version)
Generates correct, executable spatial analysis workflows from parsed queries.
"""

from typing import List, Dict, Set
from pprint import pprint
import re

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

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize feature names for safe use in layer names.
        Converts spaces, hyphens, and special characters to underscores.
        """
        # Replace spaces, hyphens, and other problematic characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's lowercase for consistency
        return sanitized.lower()

    @staticmethod
    def _add_step(workflow: List[Dict], operation: str, **kwargs):
        """Adds a new step to the workflow list."""
        step = {
            'step': len(workflow) + 1,
            'operation': operation,
            **kwargs
        }
        workflow.append(step)

    @staticmethod
    def _get_or_create_feature_layer(
        workflow: List[Dict], 
        loaded_layers: Set[str], 
        feature_type: str
    ) -> str:
        """
        Creates a filtered layer for a feature type if it doesn't already exist.
        Returns the name of the layer.
        """
        # Sanitize the feature_type for safe layer naming
        safe_feature_type = WorkflowGenerator._sanitize_name(feature_type)
        output_layer = f"layer_{safe_feature_type}"
        
        if output_layer not in loaded_layers:
            WorkflowGenerator._add_step(
                workflow,
                'filter_by_category',
                input_layer='base_landuse',
                category=feature_type,  # Use original feature_type for the actual filter
                output_layer=output_layer
            )
            loaded_layers.add(output_layer)
        return output_layer

    @staticmethod
    def _get_or_create_buffered_layer(
        workflow: List[Dict], 
        loaded_layers: Set[str],
        input_layer: str,
        distance: int
    ) -> str:
        """Creates a buffered layer if it doesn't already exist."""
        output_layer = f"{input_layer}_buffer_{distance}m"
        if output_layer not in loaded_layers:
            WorkflowGenerator._add_step(
                workflow,
                'buffer',
                input_layer=input_layer,
                distance=distance,
                output_layer=output_layer
            )
            loaded_layers.add(output_layer)
        return output_layer

    def generate_workflow(self, parsed_query: ParsedQuery) -> List[Dict]:
        """
        Generates a complete spatial analysis workflow from a parsed query.
        This method is self-contained and manages its own state.

        Args:
            parsed_query: A validated ParsedQuery object.

        Returns:
            A list of dictionary-defined steps for the GIS engine.
        """
        workflow: List[Dict] = []
        loaded_layers: Set[str] = set()

        # Step 1: Load the base OpenStreetMap data for the location.
        self._add_step(workflow, 'load_osm_data', location=parsed_query.location, output_layer='base_landuse')
        loaded_layers.add('base_landuse')

        # Step 2: Identify the initial set of candidates based on the query's TARGET.
        # This is the critical fix - we start with what the user is looking for!
        candidate_layer = self._get_or_create_feature_layer(
            workflow, loaded_layers, parsed_query.target
        )
        
        # Step 3: Sequentially apply constraints to refine the candidate layer.
        # Separate positive and negative constraints for proper logical flow
        positive_constraints = [c for c in parsed_query.constraints 
                              if c.relationship in [SpatialRelationship.WITHIN, SpatialRelationship.NEAR]]
        negative_constraints = [c for c in parsed_query.constraints 
                              if c.relationship in [SpatialRelationship.NOT_WITHIN, SpatialRelationship.FAR_FROM]]

        # Apply positive constraints (intersections) first to narrow down possibilities.
        for i, constraint in enumerate(positive_constraints):
            constraint_feature_layer = self._get_or_create_feature_layer(
                workflow, loaded_layers, constraint.feature_type
            )
            
            # If there's a distance, buffer the constraint layer first.
            if constraint.distance_meters:
                op_layer = self._get_or_create_buffered_layer(
                    workflow, loaded_layers, constraint_feature_layer, constraint.distance_meters
                )
            else:
                op_layer = constraint_feature_layer

            # Create descriptive layer name that explains what we're doing
            safe_constraint_type = self._sanitize_name(constraint.feature_type)
            constraint_desc = safe_constraint_type
            if constraint.distance_meters:
                constraint_desc += f"_buffer_{constraint.distance_meters}m"
            
            new_candidate_layer = f"candidates_intersect_{constraint_desc}_{i+1}"
            self._add_step(
                workflow,
                'intersect',
                input_layers=[candidate_layer, op_layer],
                output_layer=new_candidate_layer
            )
            candidate_layer = new_candidate_layer

        # Apply negative constraints (differences) last to carve out exclusions.
        for i, constraint in enumerate(negative_constraints):
            constraint_feature_layer = self._get_or_create_feature_layer(
                workflow, loaded_layers, constraint.feature_type
            )

            if constraint.distance_meters:
                op_layer = self._get_or_create_buffered_layer(
                    workflow, loaded_layers, constraint_feature_layer, constraint.distance_meters
                )
            else:
                op_layer = constraint_feature_layer
            
            # Create descriptive layer name for negative constraints
            safe_constraint_type = self._sanitize_name(constraint.feature_type)
            constraint_desc = safe_constraint_type
            if constraint.distance_meters:
                constraint_desc += f"_buffer_{constraint.distance_meters}m"
            
            new_candidate_layer = f"candidates_difference_{constraint_desc}_{i+1}"
            self._add_step(
                workflow,
                'difference',
                input_layers=[candidate_layer, op_layer],
                output_layer=new_candidate_layer
            )
            candidate_layer = new_candidate_layer

        # Final Step: Rename the final result layer for clarity.
        target_layer_name = f"layer_{self._sanitize_name(parsed_query.target)}"
        if candidate_layer != target_layer_name:
            self._add_step(
                workflow, 
                'rename_layer', 
                input_layer=candidate_layer, 
                output_layer='final_result'
            )

        return workflow


if __name__ == '__main__':
    # A realistic query matching the structure of the AI-GIS project.
    # Goal: "Find parks within 500m of residential areas but not within 200m of commercial zones in NYC."
    sample_query = ParsedQuery(
        target='park',  # This is what we're looking for - the target!
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

    # Test with a more complex query
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
                distance_meters=None  # No buffer - direct intersection
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