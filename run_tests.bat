@echo off
REM Batch script to run tests with venv activated

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running Task Queue Architecture Tests...
echo.

python test_task_queue.py

echo.
echo Tests complete. Press any key to exit...
pause > nul
