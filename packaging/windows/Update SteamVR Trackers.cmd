@echo off
title Kinect RGB-D - Update SteamVR trackers
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-SteamVR-Trackers.ps1"
if errorlevel 1 echo Update did not complete. Read the message above.
pause
