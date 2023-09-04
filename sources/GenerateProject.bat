@echo off
REM /*! -------------------------------------------------
REM * @file Create.bat
REM * @brief Configuration.json�������ƂɊJ�������\�z����
REM * @author nishihama takashi
REM * 
REM */ --------------------------------------------------

set ROOT_DIR=%CD%
set PROJECT_DIR=%CD%

REM �h���C�u�̃f�B���N�g�����擾
cd \
set DRIVE_DIR=%CD%
cd %ROOT_DIR%

REM .root�t�@�C����������܂Ńf�B���N�g����k��
:FindRootLoop
if exist %ROOT_DIR%\*.root (
    goto :FindRootEnd
)

REM �h���C�u�f�B���N�g���܂ők��
if not %ROOT_DIR%==%DRIVE_DIR% (
    cd ..\
    set ROOT_DIR=%CD%
    goto :FindRootLoop
)

REM ������Ȃ������̂ŏI��
echo .root not found
exit
:FindRootEnd

powershell -ExecutionPolicy RemoteSigned %ROOT_DIR%\Make\CreateBuildEnviroment -RootPath %ROOT_DIR%\ -ProjectPath %PROJECT_DIR%\ -ConfigPath %PROJECT_DIR%\Configuration.json
pause