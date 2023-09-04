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
cd ..\

set TOOL_DIR=%ROOT_DIR%\tool\tool_executer
set OUT_DIR=%ROOT_DIR%\package\model
set RESOUCE_DIR=%CD%

for /d %%d in (*) do (
    set MODEL_DIR=%RESOUCE_DIR%\%%d
    cd %RESOUCE_DIR%\%%d
    for %%f in (*.gltf) do (
       echo %RESOUCE_DIR%\%%d\%%f
       echo %OUT_DIR%\%%d
       call %TOOL_DIR%\bin\ToolExecuter.exe -i %RESOUCE_DIR%\%%d\%%f -o %OUT_DIR%\%%d
    )
    for %%f in (*.obj) do (
       echo %RESOUCE_DIR%\%%d\%%f
       echo %OUT_DIR%\%%d
       call %TOOL_DIR%\bin\ToolExecuter.exe -i %RESOUCE_DIR%\%%d\%%f -o %OUT_DIR%\%%d
    )
    cd %RESOUCE_DIR%
)
pause