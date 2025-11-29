#!/usr/bin/env python3
"""
AI Geospatial Analyst - Professional Workbench
Theme: Dark Grey / Gold Accent
Features: Split-screen, Map Visualization, Chain-of-Thought, Export
"""

import streamlit as st
import time
import json
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from datetime import datetime
import os

# Import Core Logic
from src.core.orchestrator import MasterOrchestrator
from src.core.planners.query_parser import QueryParser, QueryParserError

# ==============================================================================
# CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="AI GIS Analyst",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="collapsed"
)

# Professional Dark/Gold Theme CSS
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Golden Accents for Headers */
    h1, h2, h3 {
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    .gold-text {
        color: #D4AF37; /* Metallic Gold */
    }
    
    /* Card Styling */
    .stCard {
        background-color: #1A1C24;
        border: 1px solid #2D2F36;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Chat Bubbles */
    .user-msg {
        background-color: #262730;
        border-left: 3px solid #D4AF37;
        padding: 12px;
        border-radius: 0 8px 8px 8px;
        margin-bottom: 10px;
        color: #E0E0E0;
    }
    .agent-msg {
        background-color: #1E2029;
        border-left: 3px solid #00A2FF; /* Azure for tech */
        padding: 12px;
        border-radius: 0 8px 8px 8px;
        margin-bottom: 10px;
        color: #C0C0C0;
    }
    
    /* Chain of Thought Block */
    .cot-box {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        background-color: #111;
        color: #00D26A; /* Success Green */
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #333;
        margin-top: 5px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1A1C24;
        color: #D4AF37;
        border: 1px solid #D4AF37;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================

if "orchestrator" not in st.session_state:
    with st.spinner("⚙️ Booting AI Engines..."):
        st.session_state.orchestrator = MasterOrchestrator(max_loops=15, use_task_queue=True)
        st.session_state.query_parser = QueryParser()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_base_map(gdf):
    """
    Heavy lifting function: CRS conversion, bounds calculation, and geometry simplification.
    Not cached because folium.GeoJson with lambda style functions can't be pickled.
    """
    # 1. Coordinate System Check
    try:
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf_map = gdf.to_crs("EPSG:4326")
        else:
            gdf_map = gdf
    except:
        gdf_map = gdf

    # 2. Calculate Center
    try:
        bounds = gdf_map.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    except:
        center_lat, center_lon = 20, 0 # World view fallback

    # 3. Initialize Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True # Performance boost
    )

    # 4. Split Data Types
    points = gdf_map[gdf_map.geometry.type == 'Point']
    polygons = gdf_map[gdf_map.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    # A. OPTIMIZED POINTS: Use FastMarkerCluster
    if not points.empty:
        # Prepare raw list of [lat, lon, popup] for JS handling
        # This is much faster than creating Folium objects for every row
        locations = []
        popups = []
        for _, row in points.iterrows():
            locations.append([row.geometry.y, row.geometry.x])
            # Simplified popup text
            name = str(row.get('name', row.get('amenity', 'Feature')))[:50]
            popups.append(name)
            
        FastMarkerCluster(
            data=locations,
            popups=popups,
            name="Clustered Points"
        ).add_to(m)

    # B. OPTIMIZED POLYGONS: Simplify Geometry
    if not polygons.empty:
        # Simplify geometry (tolerance 0.0001 ~ 10 meters) reduces DOM size drastically
        polygons_simple = polygons.copy()
        polygons_simple['geometry'] = polygons.simplify(tolerance=0.0001)
        
        # Fix Datetime columns for JSON serialization
        for col in polygons_simple.columns:
            if pd.api.types.is_datetime64_any_dtype(polygons_simple[col]):
                polygons_simple[col] = polygons_simple[col].astype(str)

        folium.GeoJson(
            polygons_simple,
            name="Polygons",
            style_function=lambda x: {
                'fillColor': '#00A2FF',
                'color': '#D4AF37',
                'weight': 1,
                'fillOpacity': 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in polygons_simple.columns[:3] if c != 'geometry'],
                localize=True
            )
        ).add_to(m)

    # Fit bounds if valid
    try:
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except:
        pass
        
    return m

def render_map(gdf):
    """Lightweight wrapper that renders the cached map object."""
    if gdf is None or gdf.empty:
        st.info("No spatial data to display.")
        return

    with st.spinner("Rendering map..."):
        m = generate_base_map(gdf)
        # returned_objects=[] stops the app from reloading when you zoom/pan
        st_folium(m, width="100%", height=500, returned_objects=[])

# ==============================================================================
# UI LAYOUT
# ==============================================================================

# Top Bar
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.write("## 🛰️")
with col_title:
    st.markdown("## AI Geospatial Analyst <span class='gold-text'>PRO</span>", unsafe_allow_html=True)
st.markdown("---")

# Split Layout: Left (Logic) - Right (Visuals)
logic_col, viz_col = st.columns([0.4, 0.6], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: INTERACTION & LOGIC
# ------------------------------------------------------------------------------
with logic_col:
    st.markdown("### 💬 Query & Reasoning")
    
    # Chat History Container
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("<div style='color: #666; text-align: center; margin-top: 50px;'>Ready to analyze geospatial data.<br>Try: 'Find pharmacies near hospitals in Pune'</div>", unsafe_allow_html=True)
        
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div class='user-msg'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='agent-msg'><b>Analyst:</b> {msg}</div>", unsafe_allow_html=True)

    # Input Area
    with st.form(key="query_form"):
        user_input = st.text_input("Enter Request:", placeholder="e.g., Find schools in Berlin...")
        submit_btn = st.form_submit_button("🚀 Execute Analysis", use_container_width=True)

    # LOGIC EXECUTION
    if submit_btn and user_input:
        # Add user message to state
        st.session_state.chat_history.append(("user", user_input))
        
        with st.status("🧠 Processing Workflow...", expanded=True) as status:
            try:
                # 1. Parsing
                st.write("📝 Parsing natural language...")
                parsed = st.session_state.query_parser.parse(user_input)
                st.markdown(f"<div class='cot-box'>Target: {parsed.target}<br>Location: {parsed.location}</div>", unsafe_allow_html=True)
                
                # 2. Execution
                st.write("⚙️ Orchestrating GIS tasks...")
                start_time = time.time()
                result = st.session_state.orchestrator.run(parsed)
                duration = time.time() - start_time
                
                if result["success"]:
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                    
                    # Format success message
                    count = len(result.get("final_result", [])) if result.get("final_result") is not None else 0
                    msg = f"Analysis complete in {duration:.2f}s. Found **{count}** features."
                    st.session_state.chat_history.append(("agent", msg))
                    
                    # Save result for Right Column
                    st.session_state.last_result = result
                    st.rerun() # Force redraw to update Right Column
                else:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(result.get("error"))
                    st.session_state.chat_history.append(("agent", f"Error: {result.get('error')}"))

            except Exception as e:
                status.update(label="💥 Critical Error", state="error")
                st.error(str(e))

# ------------------------------------------------------------------------------
# RIGHT COLUMN: WORKSPACE (Visuals)
# ------------------------------------------------------------------------------
with viz_col:
    st.markdown("### 🗺️ Workspace")
    
    result = st.session_state.last_result
    
    if result:
        # Tabs for different views
        tab_map, tab_data, tab_logic = st.tabs(["📍 Map View", "📄 Data Table", "🧠 Reasoning Trace"])
        
        # 1. MAP TAB
        with tab_map:
            if result.get("final_result") is not None:
                gdf = result["final_result"]
                st.markdown(f"**Layer:** `{result.get('final_layer_name', 'Result')}` | **Count:** `{len(gdf)}`")
                render_map(gdf)
            else:
                st.info("The result was numerical (count/summary), not a map layer.")
                # For statistical queries, extract count from multiple possible sources
                count = "N/A"
                count_detail = ""
                
                # Try multiple extraction patterns from parameters_hint or error message
                if result.get("parameters_hint"):
                    reason = result["parameters_hint"].get("reason", "")
                    # Parse multiple patterns:
                    # 1. "According to OpenStreetMap, there are XXX named lakes..."
                    # 2. "The primary target 'lake' has 0 features at the specified location"
                    # 3. Error messages with counts
                    import re
                    
                    # Try pattern: "there are XXX"
                    match = re.search(r'there are (\d+)', reason)
                    if match:
                        count = match.group(1)
                    else:
                        # Try pattern: "has XXX features"
                        match = re.search(r'has (\d+) features', reason)
                        if match:
                            count = match.group(1)
                        else:
                            # Try pattern: "Found XXX" or "Found **0**"
                            match = re.search(r'Found\s+\*?\*?(\d+)\*?\*?', reason)
                            if match:
                                count = match.group(1)
                
                # Fallback: check error message
                if count == "N/A" and result.get("error"):
                    error_msg = result.get("error", "")
                    import re
                    match = re.search(r'(\d+)\s+features', error_msg)
                    if match:
                        count = match.group(1)
                
                # Last fallback: try metrics object
                if count == "N/A":
                    metrics = result.get("metrics")
                    if metrics:
                        if isinstance(metrics, dict):
                            count = metrics.get("total_count") or metrics.get("successful_actions", "N/A")
                        else:
                            count = getattr(metrics, "total_count", None) or getattr(metrics, "successful_actions", "N/A")
                
                # Check if we have split counts (named vs all)
                # For multi-target queries, sum all probe results
                probe_results = result.get("probe_results", [])
                if probe_results and len(probe_results) > 0:
                    # Multi-target: sum all probe counts
                    total_named = sum(getattr(p, 'count_named', 0) or 0 for p in probe_results)
                    total_all = sum(getattr(p, 'count_all', 0) or 0 for p in probe_results)
                    
                    # If count not extracted from reason string, use probe sum
                    if count == "N/A":
                        count = str(total_named) if total_named > 0 else str(total_all)
                    
                    # Show detailed breakdown for multi-target
                    if len(probe_results) > 1:
                        count_detail = f"Multi-target: Total Named: {total_named}, All including unnamed: {total_all}"
                        st.metric("Result Count", count, delta=count_detail)
                    else:
                        # Single target with named/all split
                        probe = probe_results[0]
                        if hasattr(probe, 'count_all') and probe.count_all is not None and probe.count_all != probe.count_named:
                            count_detail = f"(Named: {probe.count_named}, All including unnamed: {probe.count_all})"
                            st.metric("Result Count", count, delta=count_detail)
                        else:
                            st.metric("Result Count", count)
                else:
                    st.metric("Result Count", count)
                
                if count_detail:
                    st.caption(f"📊 {count_detail}")

        # 2. DATA TAB
        with tab_data:
            if result.get("final_result") is not None:
                gdf = result["final_result"]
                # Drop geometry for cleaner table view
                df_display = pd.DataFrame(gdf.drop(columns='geometry'))
                st.dataframe(df_display, height=500, use_container_width=True)
            else:
                st.info("No attribute data available.")

        # 3. LOGIC TAB
        with tab_logic:
            # Prepare download data
            export_data = {
                "query": st.session_state.chat_history[-2][1] if len(st.session_state.chat_history) > 1 else "unknown",
                "plan": result.get("action_sequence", []),
                "execution_log": result.get("execution_log", [])
            }
            
            # Download Buttons
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if result.get("final_result") is not None:
                    # Only show download if it's a spatial dataframe
                    if isinstance(result["final_result"], gpd.GeoDataFrame):
                        tmp_path = "temp_result.gpkg"
                        result["final_result"].to_file(tmp_path, driver="GPKG")
                        with open(tmp_path, "rb") as f:
                            st.download_button("📥 Download GeoPackage", f, file_name="analysis_result.gpkg")
            with col_d2:
                json_log = json.dumps(export_data, indent=2, default=str)
                st.download_button("📜 Download Workflow JSON", json_log, file_name="workflow_logic.json")

            st.markdown("#### ⛓️ Chain of Thought")
            
            # ROBUST EXTRACTION: Check all possible locations for the plan
            steps = []
            if result.get("action_sequence"):
                steps = result["action_sequence"]
            elif result.get("execution_log"):
                steps = result["execution_log"]
            
            if steps:
                for step in steps:
                    # Determine Tool Name and Details
                    tool_name = step.get('tool_name', 'Unknown Tool').upper()
                    
                    # Create a clean parameters string
                    params = step.get('parameters', {})
                    if not params:
                        # Sometimes parameters are at the top level in execution_log
                        params = {k:v for k,v in step.items() if k not in ['tool_name', 'description', 'timestamp']}
                    
                    # Filter out massive internal objects from display
                    clean_params = {k: (str(v)[:50] + "..." if len(str(v)) > 50 else v) for k,v in params.items()}
                    param_str = json.dumps(clean_params)

                    # Render Card
                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E1E; border-left: 3px solid #D4AF37; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                            <div style="color: #D4AF37; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.05em;">{tool_name}</div>
                            <div style="color: #AAAAAA; font-family: monospace; font-size: 0.8rem; margin-top: 4px;">{param_str}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.info("No detailed reasoning trace available for this run.")

    else:
        # Empty State
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; color: #444;">
                <div style="font-size: 4rem;">🗺️</div>
                <h3>No Analysis Active</h3>
                <p>Run a query on the left to generate maps and insights.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )