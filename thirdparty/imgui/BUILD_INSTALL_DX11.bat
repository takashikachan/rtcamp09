@echo off
set CURRENT_DIR=%CD%
mkdir _build
cd %CURRENT_DIR%\\_build
cmake .. -DIMGUI_WITH_BACKEND=ON -DIMGUI_BACKEND_PLATFORM=WIN32 -DIMGUI_BACKEND_DX11=ON
cmake .. -DCMAKE_INSTALL_PREFIX=../dx11
cmake --build  --target install --config Debug
cd %CURRENT_DIR%\\GL\\lib
rename imgui.lib imguiD.lib
cd %CURRENT_DIR%\\_build
cmake --build . --target install --config Release
pause