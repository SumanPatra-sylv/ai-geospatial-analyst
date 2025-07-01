# benchmark_framework.py - Fixed Version Compatible with UI

import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results."""
    task_name: str
    query: str
    success: bool
    execution_time: float
    steps_count: int
    memory_usage_mb: float
    output_quality_score: float
    reasoning_clarity_score: float
    error_handling_score: float
    workflow_validity_score: float
    timestamp: str
    errors: List[str]
    generated_workflow: List[str]
    response: str = ""  # Add response field for compatibility

class GeospatialBenchmarkSuite:
    """Comprehensive benchmarking framework for geospatial LLM agents."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define benchmark tasks - simplified for quick testing
        self.benchmark_tasks = {
            "basic_data_listing": {
                "query": "List all available data files and loaded layers",
                "expected_tools": ["ListAvailableData"],
                "expected_keywords": ["Analysis Environment Status", "Files in data directory"],
                "complexity": "low"
            },
            
            "sample_data_creation": {
                "query": "Create sample geospatial data for testing",
                "expected_tools": ["CreateSampleData"],
                "expected_keywords": ["Created sample datasets", "sample_cities.shp"],
                "complexity": "low"
            },
            
            "data_loading": {
                "query": "Load the sample cities data as 'cities' layer",
                "expected_tools": ["LoadVectorData"],
                "expected_keywords": ["Successfully loaded", "features", "cities"],
                "complexity": "medium"
            },
            
            "buffer_analysis": {
                "query": "Create a 1000-meter buffer around the cities layer and save it as 'city_buffers'",
                "expected_tools": ["CreateBuffer"],
                "expected_keywords": ["Successfully created", "buffer", "city_buffers"],
                "complexity": "medium"
            },
            
            "workflow_test": {
                "query": "Create sample data, load the cities file, create a 500m buffer, and get layer information",
                "expected_tools": ["CreateSampleData", "LoadVectorData", "CreateBuffer", "GetLayerInfo"],
                "expected_keywords": ["Created sample datasets", "Successfully loaded", "buffer", "Layer:"],
                "complexity": "high"
            }
        }
    
    def evaluate_workflow_validity(self, generated_workflow: List[str], expected_tools: List[str]) -> float:
        """Evaluate logical validity of generated workflow."""
        if not generated_workflow:
            return 0.0
            
        score = 0.0
        
        # Check if essential tools are used
        tools_used = [step.split(':')[0] if ':' in step else step for step in generated_workflow]
        if expected_tools:
            essential_coverage = len(set(expected_tools) & set(tools_used)) / len(expected_tools)
            score += essential_coverage * 0.6
        else:
            score += 0.6  # If no specific tools expected, give partial credit
        
        # Check logical sequence
        if "list_available_data" in str(generated_workflow).lower():
            score += 0.2  # Good practice: check available data
            
        # Check if load operations come before processing
        load_indicators = ["load", "create_sample_data"]
        process_indicators = ["buffer", "intersect", "filter"]
        
        load_found = any(indicator in str(generated_workflow).lower() for indicator in load_indicators)
        process_found = any(indicator in str(generated_workflow).lower() for indicator in process_indicators)
        
        if load_found and process_found:
            score += 0.2
        
        return min(score, 1.0)
    
    def evaluate_reasoning_clarity(self, reasoning_log: str) -> float:
        """Evaluate clarity and logical flow of reasoning."""
        if not reasoning_log:
            return 0.0
            
        score = 0.0
        reasoning_lower = reasoning_log.lower()
        
        # Check for clear step-by-step thinking
        step_indicators = ["step", "first", "then", "next", "finally", "thought:"]
        if any(indicator in reasoning_lower for indicator in step_indicators):
            score += 0.3
            
        # Check for explicit reasoning about tool selection
        reasoning_keywords = ["because", "since", "therefore", "need to", "will use", "should"]
        if any(keyword in reasoning_lower for keyword in reasoning_keywords):
            score += 0.3
            
        # Check for consideration of data requirements
        data_keywords = ["data", "layer", "file", "available", "loaded"]
        if any(keyword in reasoning_lower for keyword in data_keywords):
            score += 0.2
            
        # Check for error handling consideration
        error_keywords = ["error", "check", "ensure", "validate", "if"]
        if any(keyword in reasoning_lower for keyword in error_keywords):
            score += 0.2
            
        return min(score, 1.0)
    
    def evaluate_error_handling(self, execution_log: str, errors: List[str]) -> float:
        """Evaluate how well the system handled errors."""
        if not errors:
            return 1.0  # No errors is perfect
            
        score = 0.0
        
        # Check if errors were caught gracefully
        for error in errors:
            if "❌" in error:  # Properly formatted error messages
                score += 0.3
            if "Error:" in error and len(error) > 20:  # Informative error messages
                score += 0.2
                
        # Check for recovery attempts in log
        if execution_log:
            recovery_keywords = ["try", "attempting", "retrying", "alternative", "fallback"]
            if any(keyword in execution_log.lower() for keyword in recovery_keywords):
                score += 0.3
            
        return min(score / max(len(errors), 1), 1.0)
    
    def run_single_benchmark(self, agent_executor, callback_class, task_name: str, task_config: Dict) -> BenchmarkResult:
        """Run a single benchmark task."""
        print(f"🔄 Running benchmark: {task_name}")
        
        start_time = time.time()
        errors = []
        generated_workflow = []
        success = False
        response_output = ""
        reasoning_log = ""
        
        try:
            # Create callback instance
            callback = callback_class()
            
            # Execute the task
            result = agent_executor.invoke(
                {"input": task_config["query"]}, 
                {"callbacks": [callback]}
            )
            
            response_output = result.get("output", "")
            
            # Extract workflow and reasoning from callback
            if hasattr(callback, 'full_log'):
                reasoning_log = str(callback.full_log)
                # Extract workflow steps from the log
                log_lines = reasoning_log.split('\n')
                generated_workflow = [
                    line.strip() for line in log_lines 
                    if any(keyword in line.lower() for keyword in ['action:', 'tool:', 'using'])
                ]
            else:
                reasoning_log = str(result)
                generated_workflow = ["Action executed"]
            
            # Check success based on expected keywords
            expected_keywords = task_config.get("expected_keywords", [])
            if expected_keywords:
                success = any(keyword.lower() in response_output.lower() for keyword in expected_keywords)
            else:
                success = "✅" in response_output and "❌" not in response_output
            
            # Extract errors from response
            if "❌" in response_output:
                error_lines = [line.strip() for line in response_output.split('\n') if "❌" in line]
                errors.extend(error_lines)
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            errors.append(error_msg)
            reasoning_log = error_msg
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate evaluation scores
        workflow_validity_score = self.evaluate_workflow_validity(
            generated_workflow, task_config.get("expected_tools", [])
        )
        
        reasoning_clarity_score = self.evaluate_reasoning_clarity(reasoning_log)
        error_handling_score = self.evaluate_error_handling(reasoning_log, errors)
        
        # Overall output quality score
        output_quality_score = (
            workflow_validity_score * 0.4 + 
            reasoning_clarity_score * 0.3 + 
            error_handling_score * 0.3
        )
        
        # Status indication
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} | {execution_time:.2f}s | Quality: {output_quality_score:.2f}")
        
        return BenchmarkResult(
            task_name=task_name,
            query=task_config["query"],
            success=success,
            execution_time=execution_time,
            steps_count=len(generated_workflow),
            memory_usage_mb=0.0,  # Could be implemented with psutil
            output_quality_score=output_quality_score,
            reasoning_clarity_score=reasoning_clarity_score,
            error_handling_score=error_handling_score,
            workflow_validity_score=workflow_validity_score,
            timestamp=datetime.now().isoformat(),
            errors=errors,
            generated_workflow=generated_workflow,
            response=response_output
        )
    
    def run_full_benchmark_suite(self, agent_executor, callback_class) -> List[BenchmarkResult]:
        """Run the complete benchmark suite."""
        results = []
        
        print("🚀 Starting Geospatial LLM Agent Benchmark Suite")
        print("=" * 60)
        
        for task_name, task_config in self.benchmark_tasks.items():
            result = self.run_single_benchmark(agent_executor, callback_class, task_name, task_config)
            results.append(result)
        
        print(f"\n📊 Benchmark Complete: {len(results)} tests executed")
        return results
    
    def run_quick_benchmark(self, agent_executor, callback_class) -> List[BenchmarkResult]:
        """Run a smaller, faster subset of the benchmark suite."""
        results = []
        
        print("⚡ Starting Quick Benchmark Suite")
        print("=" * 40)
        
        # Run only first 3 tasks for quick testing
        quick_tasks = dict(list(self.benchmark_tasks.items())[:3])
        
        for task_name, task_config in quick_tasks.items():
            result = self.run_single_benchmark(agent_executor, callback_class, task_name, task_config)
            results.append(result)
        
        print(f"\n📊 Quick Benchmark Complete: {len(results)} tests executed")
        return results
    
    def generate_benchmark_report(self, results: List[BenchmarkResult]) -> str:
        """Generate comprehensive benchmark report."""
        if not results:
            return "📄 No benchmark results to report."
            
        report = []
        report.append("# 🎯 Geospatial LLM Agent Benchmark Report")
        report.append("=" * 50)
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Tasks:** {len(results)}")
        
        # Summary statistics
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_execution_time = np.mean([r.execution_time for r in results])
        avg_quality_score = np.mean([r.output_quality_score for r in results])
        
        report.append(f"\n## 📊 Overall Performance")
        report.append(f"- **Success Rate:** {success_rate:.1%}")
        report.append(f"- **Average Execution Time:** {avg_execution_time:.2f} seconds")
        report.append(f"- **Average Quality Score:** {avg_quality_score:.2f}/1.0")
        
        # Quality breakdown
        avg_workflow = np.mean([r.workflow_validity_score for r in results])
        avg_reasoning = np.mean([r.reasoning_clarity_score for r in results])
        avg_error_handling = np.mean([r.error_handling_score for r in results])
        
        report.append(f"\n### Quality Score Breakdown")
        report.append(f"- **Workflow Validity:** {avg_workflow:.2f}")
        report.append(f"- **Reasoning Clarity:** {avg_reasoning:.2f}")
        report.append(f"- **Error Handling:** {avg_error_handling:.2f}")
        
        # Detailed results table
        report.append(f"\n## 📋 Detailed Results")
        report.append("")
        report.append("| Status | Task | Time (s) | Quality | Query |")
        report.append("|:------:|:-----|:--------:|:-------:|:------|")
        
        for result in results:
            status_icon = "✅" if result.success else "❌"
            task_name = result.task_name.replace('_', ' ').title()
            query_short = result.query[:50] + "..." if len(result.query) > 50 else result.query
            report.append(f"| {status_icon} | {task_name} | {result.execution_time:.2f} | {result.output_quality_score:.2f} | {query_short} |")
        
        # Issues and recommendations
        failed_tasks = [r for r in results if not r.success]
        low_quality_tasks = [r for r in results if r.output_quality_score < 0.7]
        
        if failed_tasks or low_quality_tasks:
            report.append(f"\n## ⚠️ Issues and Recommendations")
            
            if failed_tasks:
                report.append(f"\n### Failed Tasks ({len(failed_tasks)})")
                for task in failed_tasks:
                    report.append(f"- **{task.task_name}**: {task.errors[0][:100] if task.errors else 'Unknown error'}...")
            
            if low_quality_tasks:
                report.append(f"\n### Low Quality Tasks ({len(low_quality_tasks)})")
                for task in low_quality_tasks:
                    issues = []
                    if task.workflow_validity_score < 0.7:
                        issues.append("workflow logic")
                    if task.reasoning_clarity_score < 0.7:
                        issues.append("reasoning clarity")
                    if task.error_handling_score < 0.7:
                        issues.append("error handling")
                    report.append(f"- **{task.task_name}**: Improve {', '.join(issues)}")
        
        # Performance insights
        fastest_task = min(results, key=lambda x: x.execution_time)
        slowest_task = max(results, key=lambda x: x.execution_time)
        best_quality = max(results, key=lambda x: x.output_quality_score)
        
        report.append(f"\n## 🔍 Performance Insights")
        report.append(f"- **Fastest Task:** {fastest_task.task_name} ({fastest_task.execution_time:.2f}s)")
        report.append(f"- **Slowest Task:** {slowest_task.task_name} ({slowest_task.execution_time:.2f}s)")
        report.append(f"- **Best Quality:** {best_quality.task_name} ({best_quality.output_quality_score:.2f})")
        
        return "\n".join(report)
    
    def save_results(self, results: List[BenchmarkResult], filename: str = None):
        """Save benchmark results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Save as JSON
        results_data = [asdict(result) for result in results]
        with open(self.results_dir / f"{filename}.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as CSV for analysis
        df = pd.DataFrame(results_data)
        df.to_csv(self.results_dir / f"{filename}.csv", index=False)
        
        # Generate and save report
        report = self.generate_benchmark_report(results)
        with open(self.results_dir / f"{filename}_report.md", 'w') as f:
            f.write(report)
        
        print(f"💾 Results saved to {self.results_dir / filename}.*")
        return f"Results saved to {self.results_dir / filename}.*"

# Compatibility function for running benchmarks
def run_benchmark_evaluation(agent_executor, callback_class):
    """Run complete benchmark evaluation."""
    benchmark_suite = GeospatialBenchmarkSuite()
    results = benchmark_suite.run_full_benchmark_suite(agent_executor, callback_class)
    benchmark_suite.save_results(results)
    
    # Generate and return summary
    report = benchmark_suite.generate_benchmark_report(results)
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    success_count = sum(1 for r in results if r.success)
    print(f"Completed: {len(results)} tasks")
    print(f"Success Rate: {success_count/len(results):.1%} ({success_count}/{len(results)})")
    print(f"Average Time: {np.mean([r.execution_time for r in results]):.2f}s")
    print(f"Average Quality: {np.mean([r.output_quality_score for r in results]):.2f}/1.0")
    
    return results