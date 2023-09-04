@echo off
set CURRENT_DIR=%CD%
mkdir glm\\_build
cd %CURRENT_DIR%\\glm\\_build
cmake .. -DCMAKE_INSTALL_PREFIX=../../
cmake --build . --target install --config Debug
cd %CURRENT_DIR%\\lib
rename glm.lib glmD.lib
cd %CURRENT_DIR%\\glm\\_build
cmake --build . --target install --config Release
pause