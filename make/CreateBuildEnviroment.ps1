#/* -------------------------------------------------
#* @file CreateBuildEnviroment.ps1
#* @brief ビルド環境構築処理が記述されたpsファイル
#* @param[in] RootPath     :ライブラリのルートパス
#* @param[in] ProjectPath  :ビルド対象のプロジェクトのパス
#* @param[in] ConfigPath   :ビルド対象のビルド情報が記述されたjsonファイルへのパス
#*/ --------------------------------------------------
Param(
    [parameter(mandatory=$true)][string]$RootPath,
    [parameter(mandatory=$true)][string]$ProjectPath,
    [parameter(mandatory=$true)][string]$ConfigPath
)

#serializerを定義
Add-Type -AssemblyName System.Web.Extensions
$serializer = New-Object System.Web.Script.Serialization.JavaScriptSerializer

#/* --------------------------------------------------
#* @brief ビルド環境を構築するルートの処理
#* @param[in] RootPath     :ライブラリのルートパス
#* @param[in] ProjectPath  :ビルド対象のプロジェクトのパス
#* @param[in] ConfigPath   :ビルド対象のビルド情報が記述されたjsonファイルへのパス
#*/ --------------------------------------------------
function CreateBuildEnviroment
{
    Param(
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$ProjectPath,
        [parameter(mandatory=$true)][string]$ConfigPath
    )

    # 引数のファイルパスの"\"を"/"に置き換え
    # CMakeのエラー対応のため
    $RootPath = $RootPath.Replace("\","/")
    $ProjectPath = $ProjectPath.Replace("\","/")
    $ConfigPath = $ConfigPath.Replace("\","/")

    #RootPathへ移動
    Set-Location -Path $RootPath
    
    #$ConfigPath(Jsonファイル)をHashTableにパース
    $ProjectConfigTxt = Get-Content -Path $ConfigPath -Encoding UTF8
    $ProjectConfig = $serializer.Deserialize($ProjectConfigTxt, [System.Collections.Hashtable])

    #WorkSpaceパスを設定
    $WorkPath = $RootPath + $ProjectConfig["WorkSpacePath"] + "/"

    #基本ビルド情報を生成し、workspaceにBuildConfig.jsonとして出力
    ."Make/CreateBuildConfig.ps1"
    $WorkConfigPath = $WorkPath + "BuildConfig.json"
    CreateBuildConfig $ProjectConfig $RootPath $ProjectPath $WorkPath $WorkConfigPath
    
    #MakeTypeに合わせてビルド環境を生成する
    $MakeConfig = $ProjectConfig["MakeConfig"]
    if($MakeConfig["MakeToolType"] -eq "CMake"){
        #CMakeによる環境構築を行う
        ."Make/Cmake/CreateBuildEnviromentCmake.ps1"
        CreateBuildEnviromentCmake $ProjectConfig $RootPath $WorkConfigPath 
    }

    
    $DocConfig = $ProjectConfig["CreateDocConfig"]
    if($DocConfig["DocType"] -eq "doxygen"){
        # Doxygenによるドキュメント生成環境構築を行う
        ."Make/Doxygen/CreateBuildDoxygen.ps1"
        CreateBuildDoxygen $DocConfig $RootPath $WorkPath $ProjectConfig["ProjectName"]
    }
    $PreMakeCommand = $ProjectConfig["PreMakeCommand"]
    if($PreMakeCommand.Length -gt 0){
        #Make前に実行するコマンドファイルを生成する
        ."make/command/CreatePreMakeCommand.ps1"
        CreatePreMakeCommand $PreMakeCommand $RootPath $WorkPath
    }

    Set-Location -Path $WorkPath

    if($PreMakeCommand.Length -gt 0){
        cmd.exe /c ".\PreMakeCommand.bat"
    }

     if($MakeConfig["MakeToolType"] -eq "CMake"){
        cmd.exe /c ".\CMakeBuild.bat"
    }
    
    if($DocConfig["DocType"] -eq "doxygen"){
        cmd.exe /c ".\DoxygenCreate.bat"
    }
}

CreateBuildEnviroment $RootPath $ProjectPath $ConfigPath