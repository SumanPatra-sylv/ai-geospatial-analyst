# backend/worker/celery_app.py

from celery import Celery
import os

# Get the Redis hostname from environment variables (set in docker-compose)
redis_host = os.getenv("REDIS_HOST", "localhost")

# Create the Celery app instance
celery = Celery(
    "geospatial_tasks",
    broker=f"redis://{redis_host}:6379/0",  # Where Celery sends task messages
    backend=f"redis://{redis_host}:6379/0", # Where Celery stores task results
    include=["backend.worker.tasks"]       # A list of modules to import when the worker starts
)

celery.conf.update(
    task_track_started=True,
)