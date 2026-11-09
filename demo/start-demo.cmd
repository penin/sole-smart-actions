@echo off
rem Double-click to launch the presentation demo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-demo.ps1" %*
