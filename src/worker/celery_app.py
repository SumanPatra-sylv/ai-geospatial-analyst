# src/worker/celery_app.py - Celery application configuration
import os
from celery import Celery

# Create Celery app instance
celery = Celery(
    'geospatial_worker',
    broker=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/0",
    backend=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/0"
)

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        'master_name': 'mymaster',
    },
    
    # Task routing and execution
    task_routes={
        'worker.tasks.run_geospatial_task': {'queue': 'geospatial'},
        'worker.tasks.cleanup_old_jobs': {'queue': 'maintenance'},
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
    
    # Monitoring and logging
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-jobs': {
            'task': 'src.worker.tasks.cleanup_old_jobs',
            'schedule': 3600.0,  # Run every hour
        },
    },
)

# Auto-discover tasks
celery.autodiscover_tasks(['src.worker'])

if __name__ == '__main__':
    celery.start()