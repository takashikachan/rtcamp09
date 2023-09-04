@echo off

echo ---------------------------------------------------------------------
echo Cuda Tool Kit�AOptix SDK����K�v�ȃw�b�_�t�@�C�����R�s�[���܂��B
echo �R�s�[��́u%CD%/ex_include�v�ł�
echo �� �s�v�ȏꍇ�͕��Ă��������B
echo ---------------------------------------------------------------------

echo 10�b��R�s�[���J�n���܂��B

timeout /nobreak 10

set CURRENT_DIR=%CD%
set EX_INCLUDE_DIR=%CURRENT_DIR%\ex_include

set EX_INCLUDE_CUDA_DIR=%EX_INCLUDE_DIR%\cuda\
set EX_INCLUDE_OPTIX_DIR=%EX_INCLUDE_DIR%\optix\
set EX_INCLUDE_SUTIL_DIR=%EX_INCLUDE_DIR%\sutil\

rem @todo ���ϐ��ɂ���B
set OPTIX_CUDA_INCLUDE_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0\SDK\cuda"
set OPTIX_SDK_INCLUDE_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0\include\"
set OPTIX_SUTIL_INCLUDE_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0\SDK\sutil\"

if exist %EX_INCLUDE_DIR% (
    rmdir /s /q %EX_INCLUDE_DIR%
)

mkdir %EX_INCLUDE_DIR%

mkdir %EX_INCLUDE_CUDA_DIR%
mkdir %EX_INCLUDE_OPTIX_DIR%
mkdir %EX_INCLUDE_SUTIL_DIR%

if exist %OPTIX_CUDA_INCLUDE_DIR% (
    echo %OPTIX_CUDA_INCLUDE_DIR% to %EX_INCLUDE_CUDA_DIR%
    xcopy /e %OPTIX_CUDA_INCLUDE_DIR% %EX_INCLUDE_CUDA_DIR%
) else (
	echo Cant Found Path : %OPTIX_CUDA_INCLUDE_DIR% 
    echo Please Install CUDA Tool Kit
)

if exist %OPTIX_SDK_INCLUDE_DIR% (
    echo %OPTIX_SDK_INCLUDE_DIR% to %EX_INCLUDE_OPTIX_DIR%
    xcopy /e %OPTIX_SDK_INCLUDE_DIR% %EX_INCLUDE_OPTIX_DIR%
) else (
	echo Cant Found Path : %OPTIX_SDK_INCLUDE_DIR%
    echo Please Install Optix
)

if exist %OPTIX_SUTIL_INCLUDE_DIR% (
    echo %OPTIX_SUTIL_INCLUDE_DIR% to %EX_INCLUDE_SUTIL_DIR%
    xcopy /e %OPTIX_SUTIL_INCLUDE_DIR% %EX_INCLUDE_SUTIL_DIR%
) else (
	echo Cant Found Path : %OPTIX_SUTIL_INCLUDE_DIR%
    echo Please Install Optix
)

cd %CURRENT_DIR%