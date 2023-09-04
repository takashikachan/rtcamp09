#/* -------------------------------------------------
#* @brief Make前実行バッチを作成
#* @param[in] PreMakeCommand   :実行するコマンド配列
#* @param[in] RootPath         :ルートパス
#* @param[in] WorkSpacePath    :workspaceのパス
#*/ --------------------------------------------------
function CreatePreMakeCommand
{
    Param(
        [parameter(mandatory=$true)]$PreMakeCommand,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkSpacePath
    )

    $CommandTxt = ""
    $TxtArray = $PreMakeCommand
    foreach ($txt in $TxtArray){
       $CommandTxt += $txt + "`n"
    }
    $CommandBatPath = $WorkSpacePath + "PreMakeCommand.bat"
    $CommandTxt | Out-File $CommandBatPath -Encoding default
}