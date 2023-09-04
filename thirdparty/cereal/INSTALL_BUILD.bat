@echo off

set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\cereal
set SOURCE_BUILD_DIR=%CURRENT_DIR%\cereal\_build
set SOURCE_INSTALL_DIR=%CURRENT_DIR%\

rmdir /s /q %SOURCE_BUILD_DIR%
mkdir %SOURCE_BUILD_DIR%

rem build tinygltf for debug
cd %SOURCE_BUILD_DIR%
cmake .. -DCMAKE_INSTALL_PREFIX=%SOURCE_INSTALL_DIR%
cmake --build . --target install --config Release
cd %CURRENT_DIR%
pause