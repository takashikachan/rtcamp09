@echo off

set ROOT_DIR=%CD%
set PROJECT_DIR=%CD%
REM ドライブのディレクトリを取得
cd \
set DRIVE_DIR=%CD%
cd %ROOT_DIR%

REM .rootファイルが見つかるまでディレクトリを遡る
:FindRootLoop
if exist %ROOT_DIR%\*.root (
    goto :FindRootEnd
)

REM ドライブディレクトリまで遡る
if not %ROOT_DIR%==%DRIVE_DIR% (
    cd ..\
    set ROOT_DIR=%CD%
    goto :FindRootLoop
)

REM 見つからなかったので終了
echo .root not found
exit
:FindRootEnd

cd %PROJECT_DIR%

set TOOL_DIR=%ROOT_DIR%\tool\nvcc_compiler\
set OUT_DIR=%ROOT_DIR%\package\program\

set INCLUDE_DIR=%ROOT_DIR%\resource\program\include\
set EX_INCLUDE_DIR=%ROOT_DIR%\resource\program\ex_include\

if exist %OUT_DIR% (
    rmdir /s /q %OUT_DIR%
)
mkdir %OUT_DIR%

cd ..\optix\

for %%a in (*.cu) do (
    call %TOOL_DIR%\bin\nvcc_compiler.exe -i %CD%\%%a -o %OUT_DIR% -include %INCLUDE_DIR% %EX_INCLUDE_DIR% -optix-ir -O3
)

cd ..\cuda\
for %%a in (*.cu) do (
    call %TOOL_DIR%\bin\nvcc_compiler.exe -i %CD%\%%a -o %OUT_DIR% -include %INCLUDE_DIR% %EX_INCLUDE_DIR% -ptx -O3
)
pause