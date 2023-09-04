set CURRENT_DIR=%CD%
set SOURCE_DIR=%CURRENT_DIR%\DirectXTex
set SOURCE_BUILD_DIR=%CURRENT_DIR%\DirectXTex\_build
set SOURCE_INSTALL_DIR=%CURRENT_DIR%\

rmdir /s /q %SOURCE_BUILD_DIR%
mkdir %SOURCE_BUILD_DIR%

cd %SOURCE_BUILD_DIR%
cmake .. -DCMAKE_INSTALL_PREFIX=%SOURCE_INSTALL_DIR%
cmake --build . --target install --config Debug

cd %SOURCE_INSTALL_DIR%\lib
rename DirectXTex.lib DirectXTexD.lib

echo %SOURCE_BUILD_DIR%
cd %SOURCE_BUILD_DIR%
cmake --build . --target install --config Release

cd %CURRENT_DIR%

xcopy %SOURCE_DIR%\DirectXTex\DDS.h %SOURCE_INSTALL_DIR%\include
pause