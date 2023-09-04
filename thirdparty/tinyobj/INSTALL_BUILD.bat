@echo off

set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\tinyobj
set SOURCE_BUILD_DIR=%CURRENT_DIR%\tinyobj\_build
set SOURCE_INSTALL_DIR=%CURRENT_DIR%\

rmdir /s /q %SOURCE_BUILD_DIR%
mkdir %SOURCE_BUILD_DIR%

rem build tinyobj for debug
cd %SOURCE_BUILD_DIR%
cmake .. -DCMAKE_INSTALL_PREFIX=%SOURCE_INSTALL_DIR%
cmake --build . --target install --config Debug

rem rename lib to separate debug and release versions
cd %SOURCE_INSTALL_DIR%\lib
rename tinyobjloader.lib tinyobjloaderD.lib

rem build tinyobj for release
echo %SOURCE_BUILD_DIR%
cd %SOURCE_BUILD_DIR%
cmake --build . --target install --config Release

cd %CURRENT_DIR%