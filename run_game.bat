@echo off
cd /d "%~dp0"
call .\venv\Scripts\activate
python play_gui.py
pause