@echo off
set FLASK_APP=app.py
set FLASK_RUN_PORT=8000
..\plant-disease-env\Scripts\python.exe -m flask run
pause
