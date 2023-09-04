#---------------------------------------------------------------------------------
# @ brief フィルタ分けの設定
# @ param[in] target_path    : フィルタ対象となるディレクトリ
# @ param[in] root_path      : プロジェクトのroot
#---------------------------------------------------------------------------------
macro(create_source_group target_path root_path)
    # 対象のディレクトリから中にあるデータをすべて収集
    file(GLOB data_list ${target_path}/*)

    #ルートディレクトリからの相対パスにする
    file(RELATIVE_PATH relative_path ${root_path} ${target_path})

    #/を\\に置換
    string(REPLACE ../ / filter_name ${relative_path})
    string(REPLACE / \\ filter_name ${filter_name})

    set(file_list )
    
    #ファイルのみを探索
    foreach(data ${data_list})
        if(NOT IS_DIRECTORY ${data})
            message(${filter_name} " in " ${data})
            LIST(APPEND file_list ${data})
        endif()
    endforeach()

    # ファイルのフィルタ分け
    source_group(${filter_name} FILES ${file_list})

    # ディレクトリリスト
    set(directory_list )

    #ディレクトリのみを探索
    foreach(data ${data_list})
        if(IS_DIRECTORY ${data})
            LIST(APPEND directory_list ${data})
        endif()
    endforeach()

    #ディレクトリのみを再帰
    foreach(directory ${directory_list})
            create_source_group(${directory} ${root_path})        
    endforeach()
endmacro()