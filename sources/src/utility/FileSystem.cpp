/**
 * @file    FileSystem.cpp
 * @brief   簡易的なファイルシステム
 */

#include "utility/FileSystem.hpp"

#include <fstream>
#include <filesystem>

#ifdef PLATFORM_WINDOWS_64
#include <Shlwapi.h>
#endif

namespace slug
{
namespace filsystem_internal
{

/**
* @brief ファイル名を列挙
* @param[in] pattern 検索時の正規表現
* @param[in] directories ディレクトリも対象とするか
* @param[in] callback コールバック
* @return 列挙数(失敗した場合は、リターンコード(負))
*/
static int32_t EnumerateFilesImpl(const char* pattern, bool directories, EnumerateCallBack callback)
{
#ifdef PLATFORM_WINDOWS_64
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(pattern, &findData);

    if (hFind == INVALID_HANDLE_VALUE) {
        if (GetLastError() == ERROR_FILE_NOT_FOUND)
            return 0;

        return static_cast<int32_t>(FileSystemCode::Failed);
    }

    int32_t numEntries = 0;

    do {
        bool isDirectory = (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        bool isDot = strcmp(findData.cFileName, ".") == 0;
        bool isDotDot = strcmp(findData.cFileName, "..") == 0;

        if ((isDirectory == directories) && !isDot && !isDotDot) {
            callback(findData.cFileName);
            ++numEntries;
        }
    } while (FindNextFileA(hFind, &findData) != 0);

    FindClose(hFind);

    return numEntries;
#endif
    return static_cast<int32_t>(FileSystemCode::NotImplement);
}
} // filsystem_internal

using namespace filsystem_internal;

Blob::Blob(void* data, size_t size)
    : m_data(data)
    , m_size(size)
{}

Blob::~Blob()
{
    if (m_data) {
        free(m_data);
        m_data = nullptr;
    }
    m_size = 0;
}

const void* Blob::data() const
{
    return m_data;
}

size_t Blob::size() const
{
    return m_size;
}

bool DefaultFileSystem::CheckDirectoryExists(const std::string& path)
{
    return std::filesystem::exists(path.c_str()) && std::filesystem::is_directory(path.c_str());
}

bool DefaultFileSystem::CheckFileExists(const std::string& path)
{
    return std::filesystem::exists(path.c_str()) && std::filesystem::is_regular_file(path.c_str());
}

std::shared_ptr<IBlob> DefaultFileSystem::ReadFile(const std::string& path)
{

    // ファイルを開けるか確認
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }

    // ファイルサイズを取得
    file.seekg(0, std::ios::end);
    uint64_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // サイズ容量を超えているか確認
    if (size > std::numeric_limits<uint64_t>::max())
    {
        ASSERT_MSG(false, "Error: file size too larges : " << path.c_str() << ", size_t : " << size);
        return nullptr;
    }

    // メモリを確保
    char* data = static_cast<char*>(::malloc(size));
    if (data == nullptr) {
        ASSERT_MSG(false, "Error: Could Not Malloc : size_t" << size);
        return nullptr;
    }

    // 読み込み
    file.read(data, size);
    if (!file.good()) {
        ASSERT_MSG(false, "Error: Cant Read File : " << path.c_str());
        return nullptr;
    }

    return std::make_shared<Blob>(data, size);
}

bool DefaultFileSystem::WriteFile(const std::string& path, const void* data, size_t size)
{
    std::filesystem::path tmp_path(path);
    std::filesystem::path dir_path = tmp_path.parent_path();
    std::filesystem::create_directory(dir_path);

    // 開けるか確認
    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // ファイル書き込み
    if (size > 0) {
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!file.good()) {
        return false;
    }

    return true;
}

int32_t DefaultFileSystem::EnumerateFiles(std::string& path, const std::vector<std::string>& extensions, EnumerateCallBack callback, bool allow_duplicates)
{

    (void)allow_duplicates;

    // 拡張子がない場合は全て列挙
    if (extensions.empty()) {
        std::string pattern = (path + "*");
        return EnumerateFilesImpl(pattern.c_str(), false, callback);
    }

    // 拡張子がある場合は、対象のファイルを列挙
    int32_t numEntries = 0;
    for (const auto& ext : extensions) {
        std::string pattern = (path + ("*" + ext));
        int result = EnumerateFilesImpl(pattern.c_str(), false, callback);

        if (result < 0)
            return result;

        numEntries += result;
    }

    return numEntries;
}

int32_t DefaultFileSystem::EnumerateDirectories(std::string& path, EnumerateCallBack callback, bool allow_duplicates)
{
    (void)allow_duplicates;

    std::string pattern = (path + "*");
    return EnumerateFilesImpl(pattern.c_str(), true, callback);
}
} // namespace slug