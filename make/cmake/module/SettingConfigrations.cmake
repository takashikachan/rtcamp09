#---------------------------------------------------------------------------------
# @ brief ビルド構成毎の設定
# @ param[in] project_name        : プロジェクト名
# @ param[in] cmake_module_path    : CMakeのモジュールディレクトリ
#---------------------------------------------------------------------------------
macro(setting_configrations project_name cmake_module_path)

    # 構成を設定
    if(CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES ${configuration_types})
    endif()

    # コンソール表示を抑制
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

    # ビルドイベントを設定
    if(${build_type} STREQUAL "execute")
        set(PRE_BUILD_COMMAND COMMAND ${pre_build_custom_command_Default})
        set(PRE_LINK_COMMAND COMMAND ${pre_link_custom_command_Default})
        set(POST_BUILD_COMMAND COMMAND ${post_build_custom_command_Default})

        # 出力ディレクトリの追加
        set_target_properties( 
            ${project_name}
            PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${output_dir_Default}"
            LIBRARY_OUTPUT_DIRECTORY "${output_dir_Default}"
            RUNTIME_OUTPUT_DIRECTORY "${output_dir_Default}"
        )

        set(CMAKE_SUPPRESS_REGENERATION true)
        set(CMAKE_BINARY_DIR ${output_dir_Default})
        set(CMAKE_CURRENT_BINARY_DIR ${output_dir_Default})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${output_dir_Default})
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${output_dir_Default})

        message("CMAKE_SOURCE_DIR : " ${CMAKE_SOURCE_DIR})
        message("CMAKE_BINARY_DIR : " ${CMAKE_BINARY_DIR})
        message("CMAKE_CURRENT_SOURCE_DIR :" ${CMAKE_CURRENT_SOURCE_DIR})
        message("CMAKE_CURRENT_BINARY_DIR : "  ${CMAKE_CURRENT_BINARY_DIR})
        message("CMAKE_LIBRARY_OUTPUT_DIRECTORY : " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        message("CMAKE_RUNTIME_OUTPUT_DIRECTORY : " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

        foreach(CONFIGRATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
            if(pre_build_custom_command_${CONFIGRATION_TYPE})
                list(APPEND PRE_BUILD_COMMAND COMMAND if "$(Configuration)" == "${CONFIGRATION_TYPE}" (${pre_build_custom_command_${CONFIGRATION_TYPE}}))
            endif()
            
            if(pre_link_custom_command_${CONFIGRATION_TYPE})
                list(APPEND PRE_LINK_COMMAND COMMAND if "$(Configuration)" == "${CONFIGRATION_TYPE}" (${pre_link_custom_command_${CONFIGRATION_TYPE}}))
            endif()

            if(post_build_custom_command_${CONFIGRATION_TYPE})
                list(APPEND POST_BUILD_COMMAND COMMAND if "$(Configuration)" == "${CONFIGRATION_TYPE}" (${post_build_custom_command_${CONFIGRATION_TYPE}}))
            endif()
        endforeach()

        if(PRE_BUILD_COMMAND)
            add_custom_command(TARGET ${project_name} PRE_BUILD ${PRE_BUILD_COMMAND})
        endif()
        
        if(PRE_LINK_COMMAND)
            add_custom_command(TARGET ${project_name} PRE_LINK ${PRE_LINK_COMMAND})
        endif()

        if(POST_BUILD_COMMAND)
            add_custom_command(TARGET ${project_name} POST_BUILD ${POST_BUILD_COMMAND})
        endif()
    endif()

    file(RELATIVE_PATH folder ${root_path}Sources ${project_root_path}/../)
    message("FOLDER_DIR : " ${folder})
    SET_TARGET_PROPERTIES(${project_name} PROPERTIES FOLDER ${folder})

    # ビルド構成毎にプリプロセッサマクロとコンパイラオプションを設定
    foreach(CONFIGRATION_TYPE ${CMAKE_CONFIGURATION_TYPES})

        # 共通設定で初期化
        set(COMPILE_DEFINATIONS ${compile_defination_Default})
        set(COMPILE_FEATURES ${compile_features_Default})
        set(COMPILE_OPTIONS ${compile_option_Default})
        set(LINKER_OPTIONS ${linker_option_Default})
        set(LINK_PATHS ${link_paths_Default})
            
        # ビルド構成に応じて追加
        list(APPEND COMPILE_DEFINATIONS ${compile_defination_${CONFIGRATION_TYPE}})
        list(APPEND COMPILE_OPTIONS ${compile_option_${CONFIGRATION_TYPE}})
        # list(APPEND LINKER_OPTIONS ${linker_option_${CONFIGRATION_TYPE}})
        list(APPEND COMPILE_FEATURES ${compile_features_${CONFIGRATION_TYPE}})
        list(APPEND LINK_PATHS ${link_paths_${CONFIGRATION_TYPE}})

        # デバッグ用にメッセージを追加
        message(${CONFIGRATION_TYPE} " [defination]:" ${COMPILE_DEFINATIONS})
        message(${CONFIGRATION_TYPE} " [option]: " ${COMPILE_OPTIONS})
        message(${CONFIGRATION_TYPE} " [output_dir]: " ${output_dir_${CONFIGRATION_TYPE}})
        message(${CONFIGRATION_TYPE} " [features]: " ${COMPILE_FEATURES})
        message(${CONFIGRATION_TYPE} " [link_path]: " ${LINK_PATHS})

        # プリプロセッサマクロを${project_name}に設定
        target_compile_definitions(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${COMPILE_DEFINATIONS}>)

        # コンパイルオプションを${project_name}に設定
        target_compile_options(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${COMPILE_OPTIONS}>)

        #リンカーオプションを設定
        # target_link_options(${project_name} PUBLIC$<$<CONFIG:${CONFIGRATION_TYPE}>:${LINKER_OPTIONS}>)

        target_compile_features(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${COMPILE_FEATURES}>)
        
        target_link_libraries(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${LINK_PATHS}>)
        
        #依存しているライブラリのリンク設定
        foreach(library_path ${library_paths})
            string(REPLACE @Configuration ${CONFIGRATION_TYPE} link_name ${library_path})
            message("target_link_libraries " ${project_name} " in " ${link_name})
            target_link_libraries(${project_name} PUBLIC $<$<CONFIG:${CONFIGRATION_TYPE}>:${link_name}>)
        endforeach()
    endforeach(CONFIGRATION_TYPE)
endmacro()