@echo off

echo ---------------------------------------------------------------------
echo Cuda Tool Kit、Optix SDKから必要なヘッダファイルをコピーします。
echo コピー先は「%CD%/ex_include」です
echo ※ 不要な場合は閉じてください。
echo ---------------------------------------------------------------------

echo 10秒後コピーを開始します。

timeout /nobreak 10

set CURRENT_DIR=%CD%
set EX_INCLUDE_DIR=%CURRENT_DIR%\ex_include

set EX_INCLUDE_CUDA_DIR=%EX_INCLUDE_DIR%\cuda\
set EX_INCLUDE_OPTIX_DIR=%EX_INCLUDE_DIR%\optix\
set EX_INCLUDE_SUTIL_DIR=%EX_INCLUDE_DIR%\sutil\

rem @todo 環境変数にする。
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