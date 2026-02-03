@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
call conda activate CaMoE
echo.
echo ✅ VS2022 x64 + CaMoE Ready
echo.