@echo off
title Kinect RGB-D - Remove SteamVR trackers
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-SteamVR-Trackers.ps1" -Remove
if errorlevel 1 echo Removal did not complete. Read the message above.
pause
