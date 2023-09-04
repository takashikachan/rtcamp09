@echo off
set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\polynomial_optics
set SOURCE_INCLUDE_DIR=%CURRENT_DIR%\include\polynomial_optics\

cd %SOURCE_DIR%

xcopy  "%CD%" "%SOURCE_INCLUDE_DIR%" /E /I /H /Y
xcopy  "%CD%" "%SOURCE_INCLUDE_DIR%" /E /I /H /Y
xcopy  "%CD%" "%SOURCE_INCLUDE_DIR%" /E /I /H /Y

cd CURRENT_DIR