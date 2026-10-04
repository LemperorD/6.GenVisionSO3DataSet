@echo off
REM Set the path of the RflySim tools
if not defined PSP_PATH (
    SET PSP_PATH=D:\PX4PSP
    SET PSP_PATH_LINUX=/mnt/d/PX4PSP
)
cd /d %PSP_PATH%\VcXsrv
tasklist|find /i "vcxsrv.exe" >nul || Xlaunch.exe -run config1.xlaunch

cd /d "%~dp0"
wsl -d RflySim-20.04 

