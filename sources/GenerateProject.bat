@echo off
REM /*! -------------------------------------------------
REM * @file Create.bat
REM * @brief Configuration.json情報をもとに開発環境を構築する
REM * @author nishihama takashi
REM * 
REM */ --------------------------------------------------

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

powershell -ExecutionPolicy RemoteSigned %ROOT_DIR%\Make\CreateBuildEnviroment -RootPath %ROOT_DIR%\ -ProjectPath %PROJECT_DIR%\ -ConfigPath %PROJECT_DIR%\Configuration.json
pause