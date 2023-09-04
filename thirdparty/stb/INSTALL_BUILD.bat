@echo off
set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\stb
set SOURCE_INCLUDE_DIR=%CURRENT_DIR%\include\stb

rmdir /s /q %SOURCE_INCLUDE_DIR%
mkdir %SOURCE_INCLUDE_DIR%
cd %SOURCE_DIR%

for %%a in (*.h) do (
    echo xopy: %CD%\%%a to %SOURCE_INCLUDE_DIR%
    copy %CD%\%%a %SOURCE_INCLUDE_DIR% /y
)

for %%a in (*.hpp) do (
    echo xopy: %CD%\%%a to %SOURCE_INCLUDE_DIR%
    copy %CD%\%%a %SOURCE_INCLUDE_DIR% /y
)