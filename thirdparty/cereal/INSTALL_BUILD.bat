@echo off
set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\cereal
set SOURCE_INCLUDE_DIR=%CURRENT_DIR%\include\

cd %SOURCE_DIR%

xcopy  "%CD%\include" "%SOURCE_INCLUDE_DIR%" /E /I /H /Y
cd CURRENT_DIR