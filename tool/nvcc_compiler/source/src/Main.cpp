/**
 * @file    Main.cpp
 */

#include <Windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// nvccが「.」でパス分割をするらしく、上手くコンパイルできないので一旦無効
#define ENABLE_INCLUDE_OPTIX_CUDA_SDK 0

static const std::string c_nvcc_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\";
static const std::string c_nvcc_name = "nvcc.exe";

#if ENABLE_INCLUDE_OPTIX_CUDA_SDK
static const std::string cuda_toolkit_include = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\include\\";
static const std::string optix_sdk_include = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.7.0\\include\\";
static const std::string optix_sutil_include = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.7.0\\SDK\sutil\\";
#endif

/**
 * @brief stringからwstringへの変換
*/
std::wstring StringToWString(const std::string& input)
{
    size_t i;
    wchar_t* buffer = new wchar_t[input.size() + 1];
    mbstowcs_s(&i, buffer, input.size() + 1, input.c_str(), _TRUNCATE);
    std::wstring result = buffer;
    delete[](buffer); 
    buffer = nullptr;
    return result;
}

/**
 * @brief wstringからstringへの変換
*/
std::string WStringToString(const std::wstring& input)
{
    size_t i;
    char* buffer = new char[input.size() * MB_CUR_MAX + 1];
    wcstombs_s(&i, buffer, input.size() * MB_CUR_MAX + 1, input.c_str(), _TRUNCATE);
    std::string result = buffer;
    delete[](buffer);
    buffer = nullptr;
    return result;
}

 /**
 * @brief コマンドライン情報
*/
struct ArgumentConfig 
{
    std::string input_file = {};
    std::string output_dir = {};
    std::string tmp_dir = {};
    std::vector<std::string> include_paths = {};
    std::vector<std::string> compile_options = {};
    bool enable_debug = false;
};

/*
* @brief コマンドライン引き数を解析
* @param[in] argc コマンドライン引き数の数
* @param[in] argv コマンドライン引き数
* @param[in] result 解析したコマンドライン引き数情報
*/
void ParseArgument(int argc, char** argv, ArgumentConfig& result)
{
    for (auto i = 0; i < argc; ++i)
    {
        if (_stricmp(argv[i], "-i") == 0)
        {
            i++;
            result.input_file = argv[i];
        }
        else if (_stricmp(argv[i], "-o") == 0)
        {
            i++;
            result.output_dir = argv[i];
        }
        else if (_stricmp(argv[i], "-t") == 0)
        {
            i++;
            result.tmp_dir = argv[i];
        }
        else if (_stricmp(argv[i], "-include") == 0)
        {
            do 
            {
                i++;
                result.include_paths.push_back(argv[i]);
            } while (argv[i + 1][0] != '-');
        }
        else if (argv[i][0] == '-')
        {
            result.compile_options.push_back(argv[i]);
        }
    }
}

/**
 * @brief NVCC用のコマンドを生成
*/
std::string GenerateNVCCCommand(const ArgumentConfig& arg_config)
{
    std::string command = c_nvcc_path + c_nvcc_name;
    command += " " + arg_config.input_file;
    for (auto& include_path : arg_config.include_paths) {
        command += " --include-path " + include_path;
    }

#if ENABLE_INCLUDE_OPTIX_CUDA_SDK
    command += " --include-path " + cuda_toolkit_include;
    command += " --include-path " + optix_sdk_include;
    command += " --include-path " + optix_sutil_include;
#endif
    command += " --output-directory " + arg_config.output_dir;
#if 0
    if (!arg_config.tmp_dir.empty()) 
    {
        command += " --objdir-as-tempdir " + arg_config.tmp_dir;
    }
#endif

    for (auto& option : arg_config.compile_options) 
    {
        command += " " + option;
    }
    return command;
}

/**
 * @brief プロセスを起動
 * @param command コマンド
 * @param wait 起動終了を待つか
*/
bool ExecuteProcess(const char* command, bool wait) 
{
    TCHAR Buffer[MAX_PATH];
    ::GetCurrentDirectory(MAX_PATH, Buffer);

    std::wstring w_str = StringToWString(c_nvcc_path);    
    ::SetCurrentDirectory(w_str.c_str());

    STARTUPINFOA        startup_info = {};
    PROCESS_INFORMATION process_info = {};

    DWORD flag = NORMAL_PRIORITY_CLASS;
    startup_info.cb = sizeof(STARTUPINFOA);

    auto ret = ::CreateProcessA(
        nullptr,
        const_cast<char*>(command),
        nullptr,
        nullptr,
        FALSE,
        flag,
        nullptr,
        nullptr,
        &startup_info,
        &process_info);

    ::SetCurrentDirectory(Buffer);

    if (ret == 0)
    {
        fprintf_s(stderr, "Error : Failed Execute Process. Command : %s\n", command);
        ::CloseHandle(process_info.hProcess);
        ::CloseHandle(process_info.hThread);
        return false;
    }

    if (wait)
    {
        ::WaitForSingleObject(process_info.hProcess, INFINITE);
    }

    ::CloseHandle(process_info.hProcess);
    ::CloseHandle(process_info.hThread);

    return true;
}

/**
 * @brief エントリポイント
*/
int main(int argc,char* argv[])
{
    if (argc <= 0) 
    {
        printf("Error Please Input Command Line : -i(input_file) -o(output_dir) -include(include path) -(nvcc compile option)\n");
        return -1;
    }
  
    ArgumentConfig arg_config = {};
    ParseArgument(argc, argv, arg_config);

    std::string command = GenerateNVCCCommand(arg_config);
    
    if (command.empty()) 
    {
        printf("Error Cant Generate Command.\n");
        return -1;
    }

    printf("Execute Process : %s\n", command.c_str());
    if (!ExecuteProcess(command.c_str(), true))
    {
        printf("Error : Execute Process");
        return -1;
    }
    return 0;

}