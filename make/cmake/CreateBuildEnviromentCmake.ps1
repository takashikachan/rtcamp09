#/* -------------------------------------------------
#* @file CreateBuildEnviromentCmake.ps1
#* @brief cmakeによる環境構築処理が記述されたpsファイル
#* @author nishihama takashi
#*/ --------------------------------------------------

$CMakeConfigTable = [ordered]@{}

#/* --------------------------------------------------
#* @brief CMakeの変数定義文字列を生成する
#* @param[in] $key        :宣言する種別(この場合はsetが入る)
#* @param[in] $value      :定義する変数名と値が記述されたhashtable
#*/ --------------------------------------------------
function CreateVariableString{
    Param(
        [parameter(mandatory=$true)][string]$key,
        [parameter(mandatory=$true)][hashtable]$value
    )

    # 下記形式で文字列を生成し返す
    # set(変数名 値1 値2...)
    $CMakeListTxt = ""
    foreach($value_key in $value.Keys){
        $CMakeListTxt += $Key
        $CMakeListTxt += "("
        $CMakeListTxt += $value_key
        foreach($value_value in $value[$value_key]){        
            $CMakeListTxt += " "
            if($value_value.Contains(" "))
            {
                $CMakeListTxt += "`"" + $value_value + "`""
            }
            else
            {
                $CMakeListTxt += "$value_value"
            }
        }
        $CMakeListTxt += ")"
        
        # 改行を挿入
        $CMakeListTxt += "`n`r"
    }

    return $CMakeListTxt
}

#/* --------------------------------------------------
#* @brief CMakeのinclude定義文字列を生成する
#* @param[in] $key        :宣言する種別(この場合はincludeが入る)
#* @param[in] $value      :includeする値が入った配列
#*/ --------------------------------------------------
function CreateIncludeString{
    Param(
        [parameter(mandatory=$true)][string]$key,
        [parameter(mandatory=$true)]$value
    )
    # 下記形式で文字列を生成し返す
    # include(値1)
    $CMakeListTxt = ""
    foreach($value_value in $value){
        $CMakeListTxt += $Key
        $CMakeListTxt += "("
        $CMakeListTxt += $value_value
        $CMakeListTxt += ")"
        $CMakeListTxt += "`n`r"
    }
    return $CMakeListTxt
}

#/* --------------------------------------------------
#* @brief CMakeのproject定義文字列を生成する
#* @param[in] $key        :宣言する種別(この場合はprojectが入る)
#* @param[in] $value      :project名
#*/ --------------------------------------------------
function CreateProjectString{
    Param(
        [parameter(mandatory=$true)][string]$key,
        [parameter(mandatory=$true)]$value
    )
    # 下記形式で文字列を生成し返す
    # include(値1)
    $CMakeListTxt = ""
    foreach($value_value in $value){
        $CMakeListTxt += $Key
        $CMakeListTxt += "("
        $CMakeListTxt += $value_value
        $CMakeListTxt += ")"
        $CMakeListTxt += "`n`r"
    }
    return $CMakeListTxt
}

#/* --------------------------------------------------
#* @brief CMakeのmacro定義文字列を生成する
#* @param[in] $key        :宣言する種別(この場合はmacroが入る)
#* @param[in] $value      :呼び出すmacro名が入った配列
#*/ --------------------------------------------------
function CreateMacroCallString{
    Param(
        [parameter(mandatory=$true)][string]$key,
        [parameter(mandatory=$true)]$value
    )
    # 下記形式で文字列を生成し返す
    # macro(値1)
    $CMakeListTxt = ""
    foreach($value_value in $value){
        $CMakeListTxt += $value
        $CMakeListTxt += "`n`r"
    }
    return $CMakeListTxt
}

#/* --------------------------------------------------
#* @brief 構成毎のビルドオプションを生成する
#* @param[in] $MakeOptionTable      :Make時のオプション設定テーブル
#*/ --------------------------------------------------
function GenerateCMakeCompileOption{
    Param(
        [parameter(mandatory=$true)]$MakeOptionTable
    )
    $LinkPath = $MakeOptionTable["LinkPath"]
    $BuildOption = $MakeOptionTable["BuildOption"]
    $Configrations = $MakeOptionTable["ConfigrationTypes"]

    $build_option_value = $BuildOption["Default"]
    
    $key = "output_dir_Default"
    $CMakeConfigTable["set"].add($key, $build_option_value["OutPutDir"])

    $key = "compile_defination_Default"
    $CMakeConfigTable["set"].add($key,$build_option_value["DefineMacro"])

    $key = "compile_option_Default"
    $CMakeConfigTable["set"].add($key,$build_option_value["CompileOption"])

    $key = "linker_option_Default"
    $CMakeConfigTable["set"].add($key,$build_option_value["LinkerOption"])

    $key = "pre_build_custom_command_Default"
    $CMakeConfigTable["set"].add($key, $build_option_value["PreBuildCutomCommand"])

    $key = "pre_link_custom_command_Default"
    $CMakeConfigTable["set"].add($key, $build_option_value["PreLinkCutomCommand"])

    $key = "post_build_custom_command_Default"
    $CMakeConfigTable["set"].add($key, $build_option_value["PostBuildCutomCommand"])
        
    $key = "compile_features_Default"
    $CMakeConfigTable["set"].add($key, $build_option_value["CompileFeature"])

    $key = "link_paths_Default"
    $CMakeConfigTable["set"].add($key, $LinkPath["Default"])

    foreach($configration in $Configrations){
        $build_option_value = $BuildOption[$configration]

        $key = "output_dir_" + $configration
        $CMakeConfigTable["set"].add($key, $build_option_value["OutPutDir"])

        $key = "compile_defination_" + $configration
        $CMakeConfigTable["set"].add($key,$build_option_value["DefineMacro"])

        $key = "compile_option_" + $configration
        $CMakeConfigTable["set"].add($key,$build_option_value["CompileOption"])

        $key = "linker_option_" + $configration
        $CMakeConfigTable["set"].add($key,$build_option_value["LinkerOption"])

        $key = "pre_build_custom_command_" + $configration
        $CMakeConfigTable["set"].add($key, $build_option_value["PreBuildCutomCommand"])

        $key = "pre_link_custom_command_" + $configration
        $CMakeConfigTable["set"].add($key, $build_option_value["PreLinkCutomCommand"])

        $key = "post_build_custom_command_" + $configration
        $CMakeConfigTable["set"].add($key, $build_option_value["PostBuildCutomCommand"])
        
        $key = "compile_features_" + $configration
        $CMakeConfigTable["set"].add($key, $build_option_value["CompileFeature"])

        $key = "link_paths_" + $configration
        $CMakeConfigTable["set"].add($key, $LinkPath[$configration])
    }
}

#/* --------------------------------------------------
#* @brief CMakeLists.txtを生成する
#* @param[in] RootPath              :ライブラリのルートパス
#* @param[in] ProjectName           :ビルド対象のプロジェクト名
#* @param[in] BuildType             :ビルド対象のビルド種別
#* @param[in] IncludeDirectories    :ビルド対象のIncludeディレクトリ配列
#* @param[in] SourceDirectories     :ビルド対象のコンパイルするディレクトリ配列
#* @param[in] WorkSpacePath         :ビルド対象のWorkSpace
#* @param[in] Libraries             :ビルド対象が依存しているライブラリ配列
#* @param[in] ProjectRootPath       :ビルド対象のソースがあるルートパス
#* @param[in] SubDirectoryPaths     :ビルド対象がもつサブディレクトリ配列
#* @param[in] MakeOptionTable       :MakeOption
#* @param[in] MakeType              :Makeするときの種別(GoogleTest or Default or GoogleBenchMark)
#* @detail 生成したBuildConfig.json情報から各プロジェクトのビルド情報を
#*         引数としてもらいそれらをもとにCMakeLists.txtを生成する
#*
#*/ --------------------------------------------------
function CreateCMakeListText{
    Param(
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$ProjectName,
        [parameter(mandatory=$true)][string]$BuildType,
        [parameter(mandatory=$true)]$IncludeDirectories,
        [parameter(mandatory=$true)]$SourceDirectories,
        [parameter(mandatory=$true)]$WorkSpacePath,
        [parameter(mandatory=$true)]$Libraries,
        [parameter(mandatory=$true)]$ProjectRootPath,
        [parameter(mandatory=$true)]$SubDirectoryPaths,
        [parameter(mandatory=$true)]$MakeOptionTable,
        [parameter(mandatory=$true)]$MakeType,
        [parameter(mandatory=$true)]$LibraryPaths,
        [parameter(mandatory=$true)]$LinkPath
    )
    # CMake生成用のテンプレート情報をhashtableでパース
    $CMakeConfigPath = $RootPath + "Make/Cmake/CreateCMakeListConfig.json"
    $CMakeConfigText = Get-Content -Path $CMakeConfigPath -Encoding UTF8
    $CMakeConfigTable = $serializer.Deserialize($CMakeConfigText, [System.Collections.Hashtable])
    
    # テンプレートに各種値を代入していく
    $CMakeConfigTable["set_ahead"]["root_path"] = $RootPath
    $CMakeConfigTable["set_ahead"]["project_name"] = $ProjectName
    $CMakeConfigTable["set_ahead"]["project_root_path"] = $ProjectRootPath
    $CMakeConfigTable["set_ahead"]["subdirectory_paths"] = $SubDirectoryPaths
    $CMakeConfigTable["set_ahead"]["cmake_module_path"] = $RootPath + "Make/Cmake/Module/"
    $CMakeConfigTable["set_ahead"]["source_paths"] = $SourceDirectories
    $CMakeConfigTable["set_ahead"]["include_paths"] = $IncludeDirectories
    $CMakeConfigTable["set_ahead"]["build_type"] = $BuildType
    $CMakeConfigTable["set_ahead"]["library_paths"] = $LibraryPaths
    $CMakeConfigTable["set_ahead"]["dependency_lib_names"] = $Libraries
    $CMakeConfigTable["set_ahead"]["configuration_types"] = $MakeOptionTable["ConfigrationTypes"]
    $CMakeConfigTable["set_ahead"]["workspace_dir"] = $WorkSpacePath
    $MakeOptionTable["LinkPath"] = $LinkPath
    GenerateCMakeCompileOption $MakeOptionTable

    $CMakeListTxt = ""

    foreach($Key in $CMakeConfigTable["@discription_order"])
    {
        $Value = $CMakeConfigTable[$Key]
        if($Key -eq "cmake_minimum_required"){
            $CMakeListTxt += CreateVariableString $Key $Value
        }
        elseif($Key -eq "set_ahead"){
            $CMakeListTxt += CreateVariableString "set" $Value
        }
        elseif($Key -eq "set"){
            $CMakeListTxt += CreateVariableString $Key $Value
        }
        elseif($Key -eq "include"){
            $CMakeListTxt += CreateIncludeString $Key $Value[$MakeType]
        }
        elseif($Key -eq "macro"){
            $CMakeListTxt += CreateMacroCallString $Key $Value
        }
        elseif($Key -eq "project"){
            $CMakeListTxt += CreateProjectString $Key $Value
        }
    }
    $CMakeListTxtPath = $WorkSpacePath + "/CMakeLists.txt"
    $CMakeListTxt | Out-File $CMakeListTxtPath -Encoding utf8
}

function CreateCMakeBatText{
    Param(
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)]$WorkSpacePath,
        [parameter(mandatory=$true)]$MakeOptionTable
    )
    # bat生成用のテンプレート情報をhashtableでパース
    $BatConfigPath = $RootPath + "make/cmake/CreateCMakeBat.json"
    $BatConfigText = Get-Content -Path $BatConfigPath -Encoding UTF8
    $BatConfigTable = $serializer.Deserialize($BatConfigText, [System.Collections.Hashtable])
    $BatConfigTable["generator"] = $MakeOptionTable["Generator"]
    $BatConfigTable["archetecture"] = $MakeOptionTable["Archetecture"]
    $BatConfigTable["output_dir_name"] = $MakeOptionTable["BuildDirectory"]
    $BatConfigTable["generator_tool"] = $MakeOptionTable["GenaratorTool"]
    $CommandTxt = ""
    if([string]::IsNullOrEmpty($BatConfigTable["archetecture"])){
        $CommandTxt = $BatConfigTable.command["vs2017lower"]
        foreach ($key in $BatConfigTable.Keys) {
            $replace_string = "@" + $key
            $CommandTxt = $CommandTxt.replace($replace_string,$BatConfigTable[$key])
        }
    }else{
        $CommandTxt = $BatConfigTable.command["vs2019upper"]
        foreach ($key in $BatConfigTable.Keys) {
            $replace_string = "@" + $key
            $CommandTxt = $CommandTxt.replace($replace_string,$BatConfigTable[$key])
        }
    }

    $CMakeListBatPath = $WorkSpacePath + "CMakeBuild.bat"
    $CommandTxt | Out-File $CMakeListBatPath -Encoding default

}

#/* --------------------------------------------------
#* @brief CMakeによる環境構築処理を行う
#* @param[in] $ProjectConfig  :プロジェクトのビルド情報
#* @param[in] $RootPath       :ライブラリのルートパス
#* @param[in] $WorkConfigPath :ビルド情報のファイルパス
#*
#* @detail WorkSpace内の各プロジェクトフォルダ毎にCMakeLists.txtを生成する
#*
#*/ --------------------------------------------------
function CreateBuildEnviromentCmake{
    Param(
        [parameter(mandatory=$true)][hashtable]$ProjectConfig,
        [parameter(mandatory=$true)][string]$RootPath,
        [parameter(mandatory=$true)][string]$WorkConfigPath
    )

    # ビルド情報取得

    #$BuildConfigTxt = Get-Content -Path "C:/Users/nishihamaT/Desktop/prog/slug/Workspace/BasicTriangle/BuildConfig.json" -Encoding UTF8
    #$BuildConfigTable = $serializer.Deserialize($BuildConfigText, [System.Collections.Hashtable])
    Get-ItemProperty $WorkConfigPath
    $BuildConfigTxt = Get-Content -Path $WorkConfigPath -Encoding UTF8
    $BuildConfigTable = $serializer.Deserialize($BuildConfigTxt, [System.Collections.Hashtable])

    # 各種cmakeに必要な情報を抽出
    $ProjectNameTable = $BuildConfigTable["ProjectName"]
    $BuildTypeTable = $BuildConfigTable["BuildType"]
    $WorkSpaceTable = $BuildConfigTable["WorkSpaceDirectory"]
    $IncludeDirectoryTable = $BuildConfigTable["IncludeDirectory"]
    $SourceDirectoryTable = $BuildConfigTable["SourceDirectory"]
    $LibraryTable = $BuildConfigTable["Library"]
    $ProjectRootPathTable = $BuildConfigTable["ProjectRootPath"]
    $MakeOptionTable = $ProjectConfig["MakeOption"]
    $MakeType = $BuildConfigTable["MakeType"]
    $LibraryPathTable = $BuildConfigTable["LibraryPath"]
    $LinkPathTable = $BuildConfigTable["LinkPath"]
    foreach($Key in $WorkSpaceTable.Keys)
    {
        $SubDirectoryPaths = @()
        if($key -eq $ProjectConfig["ProjectName"]){
            foreach($lib_name in $LibraryTable[$key]){
                $SubDirectoryPaths += $WorkSpaceTable[$lib_name]
            }
        }

        CreateCMakeListText `
            $RootPath `
            $ProjectNameTable[$Key] `
            $BuildTypeTable[$Key] `
            $IncludeDirectoryTable[$Key] `
            $SourceDirectoryTable[$Key] `
            $WorkSpaceTable[$Key] `
            $LibraryTable[$Key] `
            $ProjectRootPathTable[$Key] `
            $SubDirectoryPaths `
            $MakeOptionTable `
            $MakeType `
            $LibraryPathTable[$Key] `
            $LinkPathTable
    }
    $ProjectWorkSpace =  $WorkSpaceTable[$ProjectConfig["ProjectName"]]
    CreateCMakeBatText $RootPath $ProjectWorkSpace $MakeOptionTable
}