#/* -------------------------------------------------
#* @brief ドキュメント生成バッチを作成
#* @param[in] DocConfig  :Doxygenのコンフィグ情報
#* @param[in] RootPath         :ルートパス
#* @param[in] WorkSpacePath    :workspaceのパス
#* @param[in] ProjectName   　 :ProjectName
#*/ --------------------------------------------------
function CreateDocBat{
    Param(
        [parameter(mandatory=$true)][hashtable]$DocConfig,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkSpacePath,
        [parameter(mandatory=$true)][string]$ProjectName
    )
    # Doxxygen実行ファイルのテンプレートをhashtableでパース
    $DoxygenBatConfigPath = $RootPath + "make/doxygen/DoxygenBat.json"
    $DoxygenBatConfigTxt = Get-Content -Path $DoxygenBatConfigPath -Encoding UTF8
    $DoxygenBatConfigTable = $serializer.Deserialize($DoxygenBatConfigTxt, [System.Collections.Hashtable])
    $DoxygenBatConfigTable["DOC_DIRECTORY"] = $RootPath + $DocConfig["OUTPUT_DIRECTORY"]
    $DoxygenBatConfigTable["DOXYGEN_TOOL_DIR"] = $RootPath + "Tool/Doxygen"
    $DoxygenBatConfigTable["CONFIG_NAME"] = $WorkSpacePath + $ProjectName + "_doc"
    $DoxygenBatConfigTable["CONFIG_PATH"] = $WorkSpacePath + "DoxygenConfig.json"

    $DoxygenBatConfigTable["DOC_DIRECTORY"] = $DoxygenBatConfigTable["DOC_DIRECTORY"].replace("/","\")
    $DoxygenBatConfigTable["DOXYGEN_TOOL_DIR"] = $DoxygenBatConfigTable["DOXYGEN_TOOL_DIR"].replace("/","\")
    $DoxygenBatConfigTable["CONFIG_NAME"] = $DoxygenBatConfigTable["CONFIG_NAME"].replace("/","\")
    $DoxygenBatConfigTable["CONFIG_PATH"] = $DoxygenBatConfigTable["CONFIG_PATH"].replace("/","\")

    $CommandTxt = ""
    $TxtArray = $DoxygenBatConfigTable["command"]
    foreach ($txt in $TxtArray){
       $CommandTxt += $txt + "`n"
    }
    foreach ($key in $DoxygenBatConfigTable.Keys) {
        $replace_string = "@" + $key
        $CommandTxt = $CommandTxt.replace($replace_string, $DoxygenBatConfigTable[$key])
    }
    $DocBatPath = $WorkSpacePath + "DoxygenCreate.bat"
    $CommandTxt | Out-File $DocBatPath -Encoding default
}

#/* -------------------------------------------------
#* @brief ドキュメント生成情報を作成
#* @param[in] DocConfig  :Doxygenのコンフィグ情報
#* @param[in] RootPath         :ルートパス
#* @param[in] WorkSpacePath    :workspaceのパス
#*/ --------------------------------------------------
function CreateDocConfig{
    Param(
        [parameter(mandatory=$true)][hashtable]$DocConfig,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkSpacePath
    )
    
    # Doxygenのコンフィグ情報から生成に使用するJsonファイルを出力
    $DoxygenConfig = @{}

    foreach($key in $DocConfig.keys){
        if($key -ne "DocType"){
            $Value = ""
            if($key -eq "OUTPUT_DIRECTORY"){
                $value = $RootPath + $DocConfig[$key]
            }
            elseif($key -eq "INPUT"){
                 $value = $RootPath + $DocConfig[$key]
            }
            else{
                $value = $DocConfig[$key]
            }
            $DoxygenConfig.add($key, $value)
        }
    }
    
    $DocConfigPath = $WorkSpacePath + "DoxygenConfig.json"
    $DoxygenConfig | ConvertTo-Json | Out-File $DocConfigPath
}

#/* -------------------------------------------------
#* @brief Doxygenによるドキュメント生成環境を構築
#* @param[in] DocConfig  :Doxygenのコンフィグ情報
#* @param[in] RootPath         :ルートパス
#* @param[in] WorkSpacePath    :workspaceのパス
#* @param[in] ProjectName      :プロジェクト名
#*/ --------------------------------------------------
function CreateBuildDoxygen
{
    Param(
        [parameter(mandatory=$true)][hashtable]$DocConfig,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkSpacePath,
        [parameter(mandatory=$true)][string]$ProjectName
    )

    CreateDocConfig $DocConfig $RootPath $WorkSpacePath
    CreateDocBat $DocConfig $RootPath $WorkSpacePath $ProjectName
}