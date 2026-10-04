@ECHO OFF

REM Set the path of the RflySim tools
if not defined PSP_PATH (
    SET PSP_PATH=D:\PX4PSP
    SET PSP_PATH_LINUX=/mnt/d/PX4PSP
)

REM kill all applications when press a key
tasklist|find /i "CopterSim.exe" && taskkill /im "CopterSim.exe"
tasklist|find /i "QGroundControl.exe" && taskkill /f /im "QGroundControl.exe"
tasklist|find /i "RflySim3D.exe" && taskkill /f /im "RflySim3D.exe"

REM UE4Path
cd /d %PSP_PATH%\RflySim3D
start %PSP_PATH%\RflySim3D\RflySim3D.exe

pause
tasklist|find /i "RflySim3D.exe" && taskkill /f /im "RflySim3D.exe"
