#---------------------------------------------------------------------------------
# @ brief ビルド構成毎の設定
# @ param[in] project_name        : プロジェクト名
# @ param[in] DPENDENCY_LIBS_NAME : 依存しているライブラリ名
# @ param[in] cmake_module_path    : CMakeのモジュールディレクトリ
#---------------------------------------------------------------------------------
macro(setting_configrations project_name cmake_module_path)
    # Debug,Develop,Releaseをビルド構成に追加
    if(CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES Debug Develop Release)
    endif()

    # プリプロセッサマクロの定義を記述
    set(COMPILE_DEFINATIONS_COMMON )
    set(COMPILE_DEFINATIONS_DEBUG MODE_DEBUG MODE_TEST_CODE MODE_PROFILE_CODE)
    set(COMPILE_DEFINATIONS_DEVELOP MODE_DEBUG MODE_TEST_CODE MODE_PROFILE_CODE)
    set(COMPILE_DEFINATIONS_RELEASE MODE_RELEASE)

    # コンパイラオプションを記述
    set(COMPILE_OPTIONS_COMMON /fp:fast /Ob1 /Ot /Zi /W4 /WX)
    set(COMPILE_OPTIONS_DEBUG /MDd)
    set(COMPILE_OPTIONS_DEVELOP /MD)
    set(COMPILE_OPTIONS_RELEASE /MD)

    # ビルド構成毎にプリプロセッサマクロとコンパイラオプションを設定
    foreach(CONFIGRATION_TYPE ${CMAKE_CONFIGURATION_TYPES})

        # 共通設定で初期化
        set(COMPILE_DEFINATIONS ${COMPILE_DEFINATIONS_COMMON})
        set(COMPILE_OPTIONS ${COMPILE_OPTIONS_COMMON})

        # ビルド構成に応じて追加
        if(${CONFIGRATION_TYPE} MATCHES "Debug")
            list(APPEND COMPILE_DEFINATIONS ${COMPILE_DEFINATIONS_DEBUG})
            list(APPEND COMPILE_OPTIONS ${COMPILE_OPTIONS_DEBUG})
        elseif(${CONFIGRATION_TYPE} MATCHES "Develop")
            list(APPEND COMPILE_DEFINATIONS ${COMPILE_DEFINATIONS_DEVELOP})
            list(APPEND COMPILE_OPTIONS ${COMPILE_OPTIONS_DEVELOP})
        elseif(${CONFIGRATION_TYPE} MATCHES "Release")
            list(APPEND COMPILE_DEFINATIONS ${COMPILE_DEFINATIONS_RELEASE})
            list(APPEND COMPILE_OPTIONS ${COMPILE_OPTIONS_RELEASE})
        endif()
        
        list(APPEND COMPILE_OPTIONS "/wd4100")
        list(APPEND COMPILE_OPTIONS "/wd4201")
 
        # プリプロセッサマクロを${project_name}に設定
        target_compile_definitions(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${COMPILE_DEFINATIONS}>)

        # コンパイルオプションを${project_name}に設定
        target_compile_options(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${COMPILE_OPTIONS}>)

        #リンク設定
        foreach(library_path ${library_paths})
            string(REPLACE @Configuration ${CONFIGRATION_TYPE} link_name ${library_path})
            message("target_link_libraries " ${project_name} "in" ${link_name})
            target_link_libraries(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${link_name}>)
        endforeach()
    endforeach(CONFIGRATION_TYPE)
endmacro()

#/* -------------------------------------------------
#* @brief cmakeによる環境構築処理
#* @param[in] project_name          :プロジェクト名
#* @param[in] project_root_path     :プロジェクトのrootディレクトリ 
#* @param[in] source_paths          :プロジェクトのソースディレクトリ
#* @param[in] include_paths         :プロジェクトのインクルードディレクトリ
#* @param[in] extension_paths       :上記以外のディレクトリ
#* @param[in] dependency_lib_names  :依存しているライブラリ名
#* @param[in] build_type            :ビルド種別(execute.exe,static:.lib,shared:.dll)
#* @param[in] cmake_module_path     :CMakeのモジュールディレクトリ
#/* -------------------------------------------------
macro(setup_build_enviroment)

    #project作成
    message("build start " ${project_name})


    # googletestをgithubコード上から取得
    include(${cmake_module_path}/DownloadProject/DownloadProject.cmake)
    download_project(
        PROJ googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG master
        UPDATE_DISCONNECTED 1
    )

    #依存しているライブラリがある場合はサブディレクトリ追加を行う
    foreach(subdirectory ${subdirectory_paths})
        message("add subdirectory " ${subdirectory})
        add_subdirectory(${subdirectory} ${subdirectory})
    endforeach()

    # googletest用のソースもサブディレクトリ追加をする
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

    # コンパイル対象のソース
    set(compile_target_sources "")

    #source_pathsから.h,.cppファイルを収集し,compile_target_sourcesに追加
    foreach(src_path ${source_paths})
        if(src_path)
            file(GLOB_RECURSE source_codes ${src_path}/*.cpp ${src_path}/*.h)
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

    setting_configrations(${project_name} ${cmake_module_path})

    target_link_libraries(${project_name} PUBLIC gtest_main)

    include(GoogleTest)
    gtest_add_tests(TARGET ${project_name})

    message("build end " ${project_name})
endmacro()