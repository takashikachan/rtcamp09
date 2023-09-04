@echo off
setlocal enabledelayedexpansion

:: 特定の.batファイルの名前
set targetFile=INSTALL_BUILD.bat

:: カレントディレクトリとそのサブフォルダを再帰的に検索
for /r %%i in (*.bat) do (
    :: ファイル名を取得
    set "filename=%%~nxi"
    
    :: 指定した.batファイル名と一致する場合に実行
    if "!filename!" equ "!targetFile!" (
        echo run: "%%i"
        pushd "%%~dpi"
        call "%%i"
        popd
    )
)

endlocal