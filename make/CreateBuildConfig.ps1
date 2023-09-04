#/* -------------------------------------------------
#* @file CreateBuildConfig.ps1
#* @brief ビルド情報の生成処理が記述されたpsファイル
#*/ --------------------------------------------------

#/* --------------------------------------------------
#* @brief Make情報のハッシュテーブル
#* ProjectNameTable         :プロジェクト毎のプロジェクト名
#* BuildTypeTable           :プロジェクト毎のビルド種別
#* LibraryTable             :プロジェクト毎の内部依存ライブラリ情報
#* IncludeDirectoryTable    :プロジェクト毎のincludeディレクトリ情報
#* SourceDirectory          :プロジェクト毎のコンパイル対象のディレクトリ情報
#* WorkSpaceDirectoryTable  :プロジェクト毎の作業ディレクトリ情報
#* ProjectRootPathTable     :プロジェクト毎のルートパス情報
#*/ --------------------------------------------------
$ProjectNameTable = @{}
$BuildTypeTable = @{}
$LibraryTable = @{}
$IncludeDirectoryTable = @{}
$SourceDirectoryTable = @{}
$WorkSpaceDirectoryTable = @{}
$ProjectRootPathTable = @{}
$LibraryPathTable = @{}
$LinkPathTable = @{}

#/* --------------------------------------------------
#* @brief プロジェクトパスからプロジェクト名を取得
#* @param[in] ProjectPath       :対象のプロジェクトのパス
#*/ --------------------------------------------------
function GetProjectNameFromPath{
    Param(
        [parameter(mandatory=$true)][string]$ProjectPath
    )

    $begine_index = $ProjectPath.LastIndexOf("/") + 1
    if($begine_index -lt 0){
        $begine_index = 0
    }
    $end_index = $ProjectPath.Length
    return $ProjectPath.Substring($begine_index, $end_index - $begine_index)
}

#/* --------------------------------------------------
#* @brief 依存情報を収集する
#* @param[in] ProjectPath       :対象のプロジェクトのパス
#* @param[in] LibraryPaths      :依存しているライブラリのディレクトリ
#* @param[in] RootPath          :ルートパス
#*/ --------------------------------------------------
function CorrectDpendName{
    Param(
        [parameter(mandatory=$true)][string]$ProjectPath,
        [parameter(mandatory=$true)]$LibraryPaths,
        [parameter(mandatory=$true)][string]$RootPath
    )

    # ライブラリ名のみを抽出する。
    $LibraryNames = @()
    foreach($LibraryPath in $LibraryPaths){
        $LibraryNames += GetProjectNameFromPath $LibraryPath
    }

    # $LibraryTableのkeyに$ProjectNameが無ければ追加
    $ProjectName = GetProjectNameFromPath $ProjectPath    
    if(-not($LibraryTable.ContainsKey($ProjectName)))
    {
        $LibraryTable.add($ProjectName, $LibraryNames);
    }

    if(-not($ProjectRootPathTable.ContainsKey($ProjectName)))
    {
        $ProjectRootPathTable.add($ProjectName, $RootPath + $ProjectPath)
    }

    # 依存している内製ライブラリのConfig情報を取得
    # 次の依存先のライブラリを再帰で収集する
    foreach($LibraryPath in $LibraryPaths)
    {
        $LibraryConfigPath = $RootPath + $LibraryPath + "/Configuration.json"
        $LibraryConfigTxt = Get-Content -Path $LibraryConfigPath -Encoding UTF8
        $LibraryConfig = $serializer.Deserialize($LibraryConfigTxt, [System.Collections.Hashtable])
        CorrectDpendName $LibraryPath $LibraryConfig["DependLibrary"] $RootPath        
    }
}

#/* --------------------------------------------------
#* @brief 同じ名前が入ってないものを挿入する
#* @param[in]dstNames:挿入先の名前配列
#* @param[in]srcNames:挿入元の名前配列
#*/ --------------------------------------------------
function InsertDiffName
{
    Param(
        [parameter(mandatory=$true)]$dstNames,
        [parameter(mandatory=$true)]$srcNames
    )
    foreach ( $srcName in $srcNames )
    {
        $isSameName = 0
        foreach($dstName in $dstNames)
        {
            if($srcName -eq $dstName)
            {
                $isSameName = 1
                break
            }
        }
        if(-not($isSameName))
        {
            $dstNames += $srcName
        }
    }
    return $dstNames
}

#/* --------------------------------------------------
#* @brief 収集した内部ライブラリの依存情報を整理する
#* @param[in] ProjectName:整理対象のプロジェクト名
#* @detail 具体的には依存しているライブラリについて
#          さらに依存しているライブラリもkey毎に記述していく
#*/ --------------------------------------------------
function RefactorDependName{
    Param(
        [parameter(mandatory=$true)]$ProjectName
    )
    #依存している内部ライブラリ名、外部ライブラリ名を取得
    $DependLibraryPaths = $LibraryTable[$ProjectName]

    #library,thirdpartyをkeyとするハッシュテーブルを作成
    $DependTable = @{}
    
    #ハッシュテーブル内に名前が重複しないように挿入していく
    foreach ( $DependLibraryPath in $DependLibraryPaths ){
        $table = RefactorDependName $DependLibraryPath
        if($table)
        {
            $LibraryTable[$ProjectName] = InsertDiffName $DependLibraryPaths $table
        }
    }

    #自身が持つライブラリ情報を挿入
    $DependTable = $LibraryTable[$ProjectName]

    return $DependTable
}

#/* --------------------------------------------------
#* @brief LibraryTableをもとに各種ディレクトリ情報を収集する
#* @param[in] ProjectName       :ビルド対象のプロジェクト名
#* @param[in] RootPath          :ライブラリのルートパス
#* @param[in] WorkSpaceRootPath :作業ディレクトリのルートパス
#*/ --------------------------------------------------
function CorrectDirectory{
    Param(
        [parameter(mandatory=$true)][string]$ProjectName,
        [parameter(mandatory=$true)][string]$ProjectDir,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkSpaceRootPath
    )
    
    # 依存している内部ライブラリ毎に、各種ディレクトリ情報を収集する
    # include,source,library
    foreach($Key in $LibraryTable.Keys){
        #依存している内部ライブラリを取得
        $DependLibraryPaths = $LibraryTable[$Key]

        #各種ディレクトリ情報に対して各種key情報挿入
        $IncludeDirectoryTable.add($Key,"")


        # value値の一時保存変数
        $inc_array = @()

        #依存している内部ライブラリ全てにおいて、各種ディレクトリ情報を一時保存変数に挿入
        foreach($DependLibraryPath in $DependLibraryPaths){
            #各種ライブラリのビルド情報を取得
            $LibraryConfigPath = $ProjectRootPathTable[$DependLibraryPath] + "/Configuration.json"
            $LibraryConfigTxt = Get-Content -Path $LibraryConfigPath -Encoding UTF8
            $LibraryConfig = $serializer.Deserialize($LibraryConfigTxt, [System.Collections.Hashtable])

            #includeディレクトリ情報を挿入
            foreach($inc in $LibraryConfig["IncludeDirectory"])
            {
                $Path = $RootPath + $inc
                $inc_array += $Path
            }
        }

        #テーブルに挿入
        $IncludeDirectoryTable[$Key] = $inc_array
    }
    
    #各プロジェクト毎において自身が持つディレクトリ情報を収集する
    foreach($Key in $LibraryTable.Keys){

        $ProjectNameTable.add($key,"")
        $BuildTypeTable.add($key,"")
        $SourceDirectoryTable.add($Key,"")

        $inc_array = $IncludeDirectoryTable[$Key]
        $src_array = @()
        

        #ビルド情報があるjsonファイルパスを設定
        $LibraryConfigPath = $ProjectRootPathTable[$key] + "/Configuration.json"
        $LibraryConfigTxt = Get-Content -Path $LibraryConfigPath -Encoding UTF8
        $LibraryConfig = $serializer.Deserialize($LibraryConfigTxt, [System.Collections.Hashtable])

        #sourceディレクトリ情報を挿入
        foreach($inc in $LibraryConfig["IncludeDirectory"])
        {
            if($inc.Contains("@RootPath/"))
            {
                $Path = $inc.Replace("@RootPath/", $RootPath)
                $inc_array += $Path
            }
            else
            {
                $Path = $inc
                $inc_array += $Path
            }
        }

        #sourceディレクトリ情報を挿入
        foreach($src in $LibraryConfig["SourceDirectory"])
        {
            $Path = $RootPath + $src
            $src_array += $Path
        }

        $LibProjectName = $LibraryConfig["ProjectName"]
        $BuildType = $LibraryConfig["BuildType"]

        #テーブルに挿入
        $IncludeDirectoryTable[$Key] = $inc_array
        $SourceDirectoryTable[$Key] = $src_array
        $ProjectNameTable[$Key] = $LibProjectName
        $BuildTypeTable[$Key] = $BuildType
    }
}

#/* --------------------------------------------------
#* @brief 作業ディレクトリを作成する
#* @param[in] ProjectName   :ビルド対象のプロジェクト名
#* @param[in] $RootWorkPath :作業ディレクトリのルートパス
#*/ --------------------------------------------------
function CreateWorkSpace{
    Param(
        [parameter(mandatory=$true)]$ProjectName,
        [parameter(mandatory=$true)]$RootWorkPath
    )
    #ワークディレクトリがある場合削除して再作成
    if((Test-Path $RootWorkPath))
    {
        Remove-Item $RootWorkPath -Force -Recurse
    }
    mkdir $RootWorkPath

    #テーブルにパスを挿入しておく
    $WorkSpaceDirectoryTable.add($ProjectName, $RootWorkPath)

    # 依存しているライブラリのworkspaceもRootWorkPath以下に作成する
    $DependLibraryNames = $LibraryTable[$ProjectName]
    foreach($DependLibraryName in $DependLibraryNames)
    {
        
        $LibraryWorkPath = $RootWorkPath + $DependLibraryName
        mkdir $LibraryWorkPath

        #テーブルにパス挿入しておく
        $WorkSpaceDirectoryTable.add($DependLibraryName, $LibraryWorkPath)
    }
}

#/* --------------------------------------------------
#* @brief 依存しているライブラリのリンク情報を収集する
#* @param[in] $RootWorkPath :作業ディレクトリのルートパス
#*/ --------------------------------------------------
function CreateLibraryPathTable{
    Param(
        [parameter(mandatory=$true)]$RootWorkPath
    )


    foreach($Key in $LibraryTable.Keys)
    {
        $LibraryWorkPaths = @()
        foreach($DependLibraryName in $LibraryTable[$Key])
        {
            if($BuildTypeTable[$DependLibraryName] -eq "static"){
                $LibraryWorkPaths += $RootWorkPath + $DependLibraryName + "/@Configuration/" + $DependLibraryName + ".lib"
            }
        }
        $LibraryPathTable.add($Key, $LibraryWorkPaths)
    }
}

#/* --------------------------------------------------
#* @brief リンクしているパス情報を収集する。
#* @param[in] $ProjectName :プロジェクト名
#* @param[in] $RootPath :ルートパス
#*/ --------------------------------------------------
function CreateLinkPathTable{
    Param(
        [parameter(mandatory=$true)]$ProjectName,
        [parameter(mandatory=$true)]$RootPath
    )

    # 依存しているライブラリのリンクを収集する
    $LinkPathTableTmp = @{}
    foreach($Key in $LibraryTable.Keys){
        #ビルド情報があるjsonファイルパスを設定
        $LibraryConfigPath = $ProjectRootPathTable[$Key] + "/Configuration.json"
        $LibraryConfigTxt = Get-Content -Path $LibraryConfigPath -Encoding UTF8
        $LibraryConfig = $serializer.Deserialize($LibraryConfigTxt, [System.Collections.Hashtable])
        $MakeOption = $LibraryConfig["MakeOption"]
        $LinkPath = $MakeOption["LinkPath"]
        $LinkPathTablePerConf = @{}
        foreach($ConfKey in $LinkPath.Keys){
            $LinkPathsConf = @()
            foreach($path in $LinkPath[$ConfKey]){
                if(-not [string]::IsNullOrEmpty($path)){
                    $LinkPathsConf += $path.Replace("@RootPath", $RootPath)
                }
            }
            $LinkPathTablePerConf.add($ConfKey, $LinkPathsConf)
        }
        $LinkPathTableTmp.add($key, $LinkPathTablePerConf)
    }

    foreach($ConfKey in $LinkPathTableTmp[$ProjectName].Keys){
        $LinkPathTable.add($ConfKey, @())
    }

    # 収集した情報をもとにLinkPathTableに格納
    foreach($Key in $LibraryTable.Keys){
        foreach($ConfKey in $LinkPathTableTmp[$Key].Keys){
            $LinkPathTable[$ConfKey] += $LinkPathTableTmp[$Key][$ConfKey]
        }
    }
}

#/* --------------------------------------------------
#* @brief ビルド情報を生成する
#* @param[in] Config           :対象のビルド情報
#* @param[in] RootPath         :ライブラリのルートパス
#* @param[in] ProjectPath      :ビルド対象のルートパス
#* @param[in] WorkPath         :作業ディレクトリのパス
#* @param[in] WorkConfigPath   :ビルド情報の出力パス
#*
#* @detail 最終的には、下記情報を作業ディレクトリ下に$WorkConfigPath(jsonファイル)として出力する
#* LibraryTable             :プロジェクト毎の内部依存ライブラリ情報
#* IncludeDirectoryTable    :プロジェクト毎のincludeディレクトリ情報
#* SourceDirectory          :プロジェクト毎のコンパイル対象のディレクトリ情報
#* WorkSpaceDirectoryTable  :プロジェクト毎の作業ディレクトリ情報
#*
#*/ --------------------------------------------------
function CreateBuildConfig{
    Param(
        [parameter(mandatory=$true)][hashtable]$ProjectConfig,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$ProjectPath,
        [parameter(mandatory=$true)][string]$WorkPath,
        [parameter(mandatory=$true)][string]$WorkConfigPath
    )
    #依存しているライブラリを収集する
    CorrectDpendName $ProjectConfig["RelativeProjectPath"] $ProjectConfig["DependLibrary"] $RootPath
    $table = RefactorDependName $ProjectConfig["ProjectName"]

    #各プロジェクトのinclude,source,libディレクトリを収集
    CorrectDirectory $ProjectConfig["ProjectName"] $ProjectPath $RootPath $WorkPath
    
    #ワークスペースパスを設定し、作成
    CreateWorkSpace $ProjectConfig["ProjectName"] $WorkPath

    # 依存しているライブラリのリンク情報を収集
    CreateLibraryPathTable $WorkPath

    # リンクしているパス情報を収集
    CreateLinkPathTable $ProjectConfig["ProjectName"] $RootPath

    #ビルド情報をハッシュテーブルにしてまとめる
    $BuildConfig = @{}
    $BuildConfig.add("ProjectName",$ProjectNameTable)
    $BuildConfig.add("BuildType",$BuildTypeTable)
    $BuildConfig.add("Library",$LibraryTable)
    $BuildConfig.add("IncludeDirectory",$IncludeDirectoryTable)
    $BuildConfig.add("SourceDirectory",$SourceDirectoryTable)
    $BuildConfig.add("WorkSpaceDirectory",$WorkSpaceDirectoryTable)
    $BuildConfig.add("ProjectRootPath", $ProjectRootPathTable)
    $BuildConfig.add("MakeType", $ProjectConfig.MakeConfig["MakeType"])
    $BuildConfig.add("LibraryPath", $LibraryPathTable)
    $BuildConfig.add("LinkPath", $LinkPathTable)

    #ビルド情報を$WorkConfigPath(jsonファイル)として出力
    $BuildConfig | ConvertTo-Json | Out-File $WorkConfigPath
}