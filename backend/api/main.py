# backend/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any

# Create the app instance first
app = FastAPI(
    title="AI Geospatial Analyst - Main API",
    description="API for processing geospatial queries using AI workers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "Server error"
        }
    )

# Import routes after app creation to avoid circular imports
try:
    from backend.api.routes import jobs
    app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
    JOBS_ROUTER_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import jobs router: {e}")
    JOBS_ROUTER_LOADED = False

@app.get("/", tags=["Root"])
def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Geospatial Analyst API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Main API health check"""
    health_status = {
        "status": "ok",
        "api": "healthy",
        "jobs_router": "loaded" if JOBS_ROUTER_LOADED else "unavailable"
    }
    
    # If jobs router is loaded, get detailed health info
    if JOBS_ROUTER_LOADED:
        try:
            from backend.api.routes.jobs import redis_client, REDIS_AVAILABLE, get_celery_task
            
            # Check Redis
            health_status["redis"] = "healthy" if REDIS_AVAILABLE else "unavailable"
            if REDIS_AVAILABLE:
                try:
                    redis_client.ping()
                    health_status["redis"] = "healthy"
                except:
                    health_status["redis"] = "error"
            
            # Check Celery
            celery_task = get_celery_task()
            health_status["celery"] = "available" if celery_task else "unavailable"
            
        except Exception as e:
            health_status["jobs_detail_error"] = str(e)
    
    return health_status

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus-compatible metrics endpoint"""
    try:
        # Basic metrics in Prometheus format
        metrics_data = []
        
        # API status metric
        metrics_data.append('# HELP api_status API status (1=healthy, 0=unhealthy)')
        metrics_data.append('# TYPE api_status gauge')
        metrics_data.append('api_status 1')
        
        # Jobs router status
        metrics_data.append('# HELP jobs_router_loaded Jobs router loaded status (1=loaded, 0=not loaded)')
        metrics_data.append('# TYPE jobs_router_loaded gauge')
        metrics_data.append(f'jobs_router_loaded {1 if JOBS_ROUTER_LOADED else 0}')
        
        # Redis status if available
        if JOBS_ROUTER_LOADED:
            try:
                from backend.api.routes.jobs import redis_client, REDIS_AVAILABLE
                redis_status = 1 if REDIS_AVAILABLE else 0
                if REDIS_AVAILABLE:
                    try:
                        redis_client.ping()
                        redis_status = 1
                    except:
                        redis_status = 0
                
                metrics_data.append('# HELP redis_status Redis connection status (1=healthy, 0=unhealthy)')
                metrics_data.append('# TYPE redis_status gauge')
                metrics_data.append(f'redis_status {redis_status}')
            except:
                pass
        
        return '\n'.join(metrics_data)
    
    except Exception as e:
        # Return basic error metric
        return f'# API metrics error\napi_error{{error="{str(e)}"}} 1'

@app.get("/info", tags=["Info"])
def api_info():
    """Get API information and available endpoints"""
    endpoints = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            endpoints.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unknown')
            })
    
    return {
        "title": app.title,
        "description": app.description,
        "version": app.version,
        "endpoints": endpoints,
        "features": {
            "jobs_processing": JOBS_ROUTER_LOADED,
            "cors_enabled": True,
            "docs_available": True
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    print("🚀 AI Geospatial Analyst API starting up...")
    print(f"📋 Jobs router loaded: {JOBS_ROUTER_LOADED}")
    print("✅ API is ready to serve requests")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    print("🛑 AI Geospatial Analyst API shutting down...")
    
    # Close Redis connection if available
    if JOBS_ROUTER_LOADED:
        try:
            from backend.api.routes.jobs import redis_client, REDIS_AVAILABLE
            if REDIS_AVAILABLE and redis_client:
                redis_client.close()
                print("📪 Redis connection closed")
        except:
            pass
    
    print("✅ Shutdown complete")

# Add a catch-all for debugging
if os.getenv("DEBUG", "false").lower() == "true":
    @app.get("/debug/routes")
    def debug_routes():
        """Debug endpoint to see all routes"""
        routes_info = []
        for route in app.routes:
            routes_info.append({
                "path": getattr(route, 'path', 'unknown'),
                "name": getattr(route, 'name', 'unknown'),
                "methods": list(getattr(route, 'methods', [])),
            })
        return {"routes": routes_info}