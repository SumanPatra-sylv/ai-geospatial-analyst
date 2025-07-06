#!/bin/sh
set -e

# This script runs as root when the container starts.

# 1. Take ownership of mounted volumes so the app can write to them.
#    This command is safe even if the directories don't exist or if no
#    volumes are mounted.
chown -R appuser:appuser /app/data /app/celery

# 2. Drop privileges and execute the main command passed from docker-compose.
#    '$@' represents the command like 'uvicorn...' or 'celery...'
exec gosu appuser "$@"