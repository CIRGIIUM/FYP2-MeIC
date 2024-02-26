@echo off
set SCRIPT_PATH=%~dp0
set FLASK_APP=%SCRIPT_PATH%\MeIC_app.py
set FLASK_ENV=development
set FLASK_DEBUG=1
start cmd /k "flask run --port=5003"
timeout /t 5
start chrome http://127.0.0.1:5003/MeIC
