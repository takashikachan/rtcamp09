#/* -------------------------------------------------
#* @brief cmakeによる環境構築処理
#* @param[in] project_name              :プロジェクト名
#* @param[in] project_root_path         :プロジェクトのrootディレクトリ 
#* @param[in] source_paths              :プロジェクトのソースディレクトリ
#* @param[in] include_paths             :プロジェクトのインクルードディレクトリ
#* @param[in] extension_paths           :上記以外のディレクトリ
#* @param[in] ssubdirectory_paths       :サブディレクトリに追加するパス
#* @param[in] dependency_lib_names      :依存しているライブラリ名
#* @param[in] build_type                :ビルド種別(execute.exe,static:.lib,shared:.dll)
#* @param[in] cmake_module_path         :CMakeのモジュールディレクトリ
#* @param[in] configuration_types       :生成するターゲット名
#* @param[in] compile_option_Defautlt   :全ターゲット共通のコンパイルオプション
#* @param[in] compile_option_**         :任意のターゲットのコンパイルオプション
#* @param[in] compile_defination_Default:全ターゲット共通のプリプロセッサマクロ宣言
#* @param[in] compile_defination_**     :任意のターゲットのプリプロセッサマクロ宣言
#* @param[in] linker_option_Default     :全ターゲット共通のリンカーオプション
#* @param[in] linker_option_**          :任意のターゲットのリンカーオプション
#* @param[in] add_custom_command_Default:全ターゲット共通のビルドイベント
#* @param[in] add_custom_command_**     :任意のターゲットのビルドイベント
#/* -------------------------------------------------
macro(setup_build_enviroment)
    #project作成
    message("build start " ${project_name})

    # フィルタリングを有効にする。
    set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

    #依存しているライブラリがある場合はサブディレクトリ追加を行う
    foreach(subdirectory ${subdirectory_paths})
        message("add subdirectory " ${subdirectory})
        add_subdirectory(${subdirectory} ${subdirectory})
    endforeach()

    # コンパイル対象のソース
    set(compile_target_sources "")

    #source_pathsから.hpp,.cpp,.h,.cファイルを収集し,compile_target_sourcesに追加
    foreach(src_path ${source_paths})
        if(src_path)
            file(GLOB_RECURSE source_codes ${src_path}/*.cpp ${src_path}/*.hpp ${src_path}/*.h ${src_path}/*.c)
            LIST(APPEND compile_target_sources ${source_codes} )
            message("compile_target_sources correct from " ${src_path})
        endif()
    endforeach()

    #インクルードディレクトリの設定
    include_directories(${include_paths})

    #ソースをすべてプロジェクトに登録
    message("build type " ${build_type})
    if(${build_type} STREQUAL "execute")
        add_executable(${project_name} ${compile_target_sources})
    elseif(${build_type} STREQUAL "static")
        add_library(${project_name} STATIC ${compile_target_sources})
    elseif(${build_type} STREQUAL "shared")
        add_library(${project_name} SHARED ${compile_target_sources})
    endif()
    
    #ソースのフィルタ分けを行う
    include(${cmake_module_path}/CreateSourceGroup.cmake)
    message("create_srouce_group " ${project_name})
    if(source_paths)
        foreach(src_path ${source_paths})
            create_source_group(${src_path} ${project_root_path}/_build)
        endforeach()
    endif()
    if(extension_paths)
        foreach(ext_path ${extension_paths})
            create_source_group(${ext_path} ${project_root_path})
        endforeach()
    endif()
    
    #ビルドの依存関係を記述
    foreach(lib_name ${dependency_lib_names})
        message("add_dependencies " ${project_name} " by " ${lib_name})
        add_dependencies(${project_name} ${lib_name})
    endforeach()


    #ビルド構成毎の設定
    include(${cmake_module_path}/SettingConfigrations.cmake)
    setting_configrations(${project_name} ${cmake_module_path})

    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${project_name})
    message("build end " ${project_name})
endmacro()