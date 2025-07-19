#!/usr/bin/env python3
"""
MasterOrchestrator - The Think-Act-Observe loop controller for iterative GIS workflows.
Phase 2: Coordinates between WorkflowGenerator and WorkflowExecutor for intelligent, adaptive execution.
"""

import time
from typing import Dict, List, Any, Optional
import geopandas as gpd

from src.core.planners.workflow_generator import WorkflowGenerator, ToolCallIntent
from src.core.executors.workflow_executor import WorkflowExecutor  
from src.core.agents.data_scout import DataScout
from src.core.planners.query_parser import ParsedQuery


class MasterOrchestrator:
    """
    The central controller for iterative Think-Act-Observe GIS workflows.
    Coordinates between the Strategist (WorkflowGenerator) and Actor (WorkflowExecutor).
    """
    
    def __init__(self, max_loops: int = 15):
        """
        Initialize the orchestrator with its agent components.
        
        Args:
            max_loops: Maximum number of Think-Act-Observe iterations (safety limit)
        """
        # Initialize core agents
        self.data_scout = DataScout()
        self.strategist = WorkflowGenerator(data_scout=self.data_scout)
        self.executor = WorkflowExecutor(enable_reasoning_log=True)
        
        # State management
        self.conversation_history: List[str] = []
        self.data_layers: Dict[str, gpd.GeoDataFrame] = {}
        self.action_sequence: List[Dict[str, Any]] = []  # For RAG learning
        
        # Configuration
        self.max_loops = max_loops

    def run(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Execute the complete Think-Act-Observe loop for a given query.
        
        Args:
            parsed_query: The structured query from the query parser
            
        Returns:
            Dictionary containing execution results and metadata
        """
        print("🎭 === MASTER ORCHESTRATOR: Starting Think-Act-Observe Loop ===")
        print(f"🎯 Query: Find '{parsed_query.target}' in '{parsed_query.location}'")
        
        # Phase 1: Initialize context with data scouting
        print("\n🔍 Phase 1: Establishing initial context...")
        context = self.strategist.get_initial_context(parsed_query)
        
        if not context["success"]:
            return {
                "success": False,
                "error": context["error"],
                "conversation_history": self.conversation_history,
                "execution_time": 0
            }
        
        # Initialize conversation history
        self.conversation_history = [context["initial_observation"]]
        original_query = context.get("original_query", f"Find {parsed_query.target} in {parsed_query.location}")
        rag_guidance = context.get("rag_guidance", "")
        
        print("✅ Initial context established successfully")
        print(f"📊 {context['initial_observation']}")
        
        # Phase 2: Think-Act-Observe Loop
        print("\n🔄 Phase 2: Starting iterative Think-Act-Observe loop...")
        
        start_time = time.time()
        loop_count = 0
        finished = False
        
        while not finished and loop_count < self.max_loops:
            loop_count += 1
            print(f"\n--- Loop {loop_count} ---")
            
            # 🧠 THINK: Get next action from strategist
            print("🧠 THINK: Deciding next action...")
            next_action = self.strategist.get_next_action(
                history=self.conversation_history,
                original_query=original_query,
                rag_guidance=rag_guidance
            )
            
            print(f"💡 Decision: {next_action.tool_name}({next_action.parameters})")
            
            # Check for completion
            if next_action.tool_name == "finish_task":
                final_layer = next_action.parameters.get("final_layer_name")
                self.conversation_history.append(f"Action: finish_task")
                self.conversation_history.append(f"Observation: Task completed. Final result: '{final_layer}'")
                finished = True
                break
            
            # 🎬 ACT: Execute single step
            print("🎬 ACT: Executing action...")
            observation, updated_layers = self.executor.execute_single_step(
                tool_call=next_action,
                current_data_layers=self.data_layers
            )
            
            # Store successful action for RAG learning
            if observation.startswith("Success"):
                executable_action = self.strategist.translate_single_action(next_action, self._build_conceptual_map())
                self.action_sequence.append(executable_action)
            
            # 👀 OBSERVE: Update state and history
            print("👀 OBSERVE: Processing results...")
            self.conversation_history.append(f"Action: {next_action.tool_name}")
            self.conversation_history.append(f"Observation: {observation}")
            self.data_layers = updated_layers
            
            print(f"📈 Current layers: {list(self.data_layers.keys())}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Phase 3: Finalization and learning
        print(f"\n🎉 Execution completed in {execution_time:.2f}s after {loop_count} loops")
        
        # Store successful pattern for future RAG retrieval
        if finished and self.action_sequence:
            print("💾 Storing successful workflow pattern for future learning...")
            self.strategist.store_successful_pattern(original_query, self.action_sequence)
        
        # Prepare final result
        final_layer_name = None
        final_result = None
        
        if self.data_layers:
            final_layer_name = list(self.data_layers.keys())[-1]
            final_result = self.data_layers[final_layer_name]
        
        return {
            "success": finished,
            "final_layer_name": final_layer_name,
            "final_result": final_result,
            "conversation_history": self.conversation_history,
            "action_sequence": self.action_sequence,
            "execution_time": execution_time,
            "loop_count": loop_count,
            "data_layers": self.data_layers,
            "reasoning_log": self.executor.get_reasoning_log()
        }
    
    def _build_conceptual_map(self) -> Dict[str, str]:
        """
        Build mapping from conceptual names to real layer names.
        
        Returns:
            Dictionary mapping conceptual_name -> real_layer_name
        """
        conceptual_map = {}
        for real_name in self.data_layers.keys():
            # Extract conceptual name: 'schools_1' -> 'schools'
            conceptual_name = "_".join(real_name.split('_')[:-1])
            conceptual_map[conceptual_name] = real_name
        return conceptual_map
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        return {
            "total_loops": len([h for h in self.conversation_history if h.startswith("Action:")]),
            "final_layers": list(self.data_layers.keys()),
            "conversation_length": len(self.conversation_history),
            "actions_taken": len(self.action_sequence),
            "executor_summary": self.executor.get_execution_summary()
        }


# Integration test
if __name__ == '__main__':
    print("🧪 === MasterOrchestrator Integration Test ===")
    
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship
    
    # Test query: Find schools near parks in Berlin
    test_query = ParsedQuery(
        target='school',
        location='Berlin, Germany', 
        constraints=[
            SpatialConstraint(feature_type='park', relationship=SpatialRelationship.NEAR, distance_meters=500)
        ],
        summary_required=True
    )
    
    orchestrator = MasterOrchestrator(max_loops=10)
    result = orchestrator.run(test_query)
    
    print("\n📊 === FINAL RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Total Loops: {result['loop_count']}")
    
    if result['final_result'] is not None:
        print(f"Final Result: {len(result['final_result'])} features")
    
    print("\n📝 Conversation History:")
    for i, entry in enumerate(result['conversation_history'], 1):
        print(f"{i}. {entry}")
    
    print("\n✅ MasterOrchestrator test completed!")
