@echo off

REM Set the path of the RflySim tools
if not defined PSP_PATH (
    SET PSP_PATH=D:\PX4PSP
    SET PSP_PATH_LINUX=/mnt/d/PX4PSP
)

tasklist|find /i "vcxsrv.exe" >nul && taskkill /f /IM "vcxsrv.exe" >nul
tasklist|find /i "vcxsrv.exe" >nul && taskkill /f /IM "vcxsrv.exe" >nul

cd /d %PSP_PATH%\VcXsrv
Xlaunch.exe -run config.xlaunch

choice /t 1 /d y /n >nul

cd /d "%~dp0"
wsl -d RflySim-20.04 ~/StartUI.sh
choice /t 1 /d y /n >nul
tasklist|find /i "vcxsrv.exe" && taskkill /f /IM "vcxsrv.exe"
wsl --shutdown
