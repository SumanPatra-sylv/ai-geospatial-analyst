# Known Issue: Large-Scale Query Optimization Needed

## Problem
Queries for large-scale statistical data (e.g., "how many lakes in Kenya") cause the system to hang because it attempts to download thousands of unnamed features.

## Root Cause
1. **No Statistical Shortcut**: System downloads all data even when only needing a count
2. **No Senior Analyst Filters**: Probes include unnamed farm ponds, streams, etc.

## Recommended Fixes (for future PR)

### Fix 1: Statistical Shortcut in `src/core/planners/execution_planner.py`
At the start of `generate_task_queue()` method, add:
```python
# Statistical Shortcut: If only summary needed and no spatial analysis
if parsed_query.summary_required and not requirements['needs_spatial_analysis']:
    # Don't download data - just return count from probe
    probe_result = data_report.probe_results[0]  # First entity
    answer = f"Found approximately {probe_result.count:,} {parsed_query.target} in {parsed_query.location}"
    
    # Return single finish task with answer
    finish_task = self._create_statistical_finish_task(answer, probe_result.count)
    return TaskQueue(tasks=[finish_task], original_query=f"Find {parsed_query.target} in {parsed_query.location}", requirements=requirements)
```

### Fix 2: Senior Analyst Filters in `src/core/agents/data_scout.py`
In `_async_query_osm_count()` method, after line 716, add:
```python
# Senior Analyst Filter: Require named features for major types
MAJOR_FEATURES = {
    'lake': ['natural=water', 'water=lake'],
    'river': ['waterway=river'],
    'mountain': ['natural=peak']
}

needs_name_filter = False
for feature_type, required_tags in MAJOR_FEATURES.items():
    for req_tag in required_tags:
        key, value = req_tag.split('=')
        if tag_dict.get(key) == value:
            needs_name_filter = True
            logger.info(f"🔍 [Senior Analyst Filter] Detected '{feature_type}' - requiring named features only")
            break

if needs_name_filter:
    tag_filter += '["name"]'  # Only count named features
```

## Workaround for Now
For testing, avoid queries like:
- ❌ "How many lakes in Kenya" (will hang)
- ✅ "Find hospitals in Berlin" (limited scope, will work)
- ✅  "Find schools near parks in London" (constrained, will work)

## Status
- Issue documented: ✅
- Fixes identified: ✅
- Implementation: ⏳ Pending (need stable file editing)
