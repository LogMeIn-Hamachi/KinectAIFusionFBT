@echo off
title Kinect RGB-D - Install SteamVR trackers
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-SteamVR-Trackers.ps1"
if errorlevel 1 echo Installation did not complete. Read the message above.
pause
