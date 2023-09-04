@echo off

set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\tinygltf
set SOURCE_BUILD_DIR=%CURRENT_DIR%\tinygltf\_build
set SOURCE_INSTALL_DIR=%CURRENT_DIR%\

rmdir /s /q %SOURCE_BUILD_DIR%
mkdir %SOURCE_BUILD_DIR%

rem build tinygltf for debug
cd %SOURCE_BUILD_DIR%
cmake .. -DCMAKE_INSTALL_PREFIX=%SOURCE_INSTALL_DIR%
cmake --build . --target install --config Debug

rem rename lib to separate debug and release versions
cd %SOURCE_INSTALL_DIR%\lib
rename tinygltf.lib tinygltfD.lib

rem build tinygltf for release
echo %SOURCE_BUILD_DIR%
cd %SOURCE_BUILD_DIR%
cmake --build . --target install --config Release

cd %CURRENT_DIR%