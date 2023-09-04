/**
 * @file    FileSystem.hpp
 * @brief   簡易的なファイルシステム
 */
#pragma once

#include "Define.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace slug
{
class IBlob;
class DefaultFileSystem;

using EnumerateCallBack = std::function<void(std::string)>;
using IBlobPtr = std::shared_ptr<IBlob>;
using DefaultFileSystemPtr = std::shared_ptr<DefaultFileSystem>;

/**
 * @brief ファイルシステムのリターンコード
*/
enum class FileSystemCode : int8_t
{
    Success = 0,        //!< 成功
    Failed = -1,        //!< 失敗
    NotFoundPath = -2,  //!< パスが見つからなかった
    NotImplement = -3   //!< 未実装
};

 /**
 * @brief バイナリデータのインターフェース
 */
class IBlob {
public:
    virtual ~IBlob() = default;
    NO_DISCARD virtual const void* data() const = 0;
    NO_DISCARD virtual size_t size() const = 0;
    /**
    * @brief 空かどうかを判定
    */
    static bool isEmpty(IBlob const& blob)
    {
        return (blob.data() != nullptr) && (blob.size() != 0);
    }
};

/**
* @brief デフォルトのバイナリデータクラス
*/
class Blob : public IBlob {
public:
    /**
    * @brief コンストラクタ
    * @param[in] data アドレス
    * @param[in] size サイズ
    */
    Blob(void* data, size_t size);

    /**
    * @brief デストラクタ
    */
    ~Blob() override;

    /**
    * @brief データのアドレスを取得
    */
    NO_DISCARD const void* data() const override;

    /**
    * @brief データのサイズを取得
    */
    NO_DISCARD size_t size() const override;
private:
    void* m_data;       //!< データアドレス
    size_t m_size;      //!< データサイズ
};

/**
* @brief ファイルシステムのインターフェース
*/
class IFileSystem {
public:
    virtual ~IFileSystem() = default;
    virtual bool CheckDirectoryExists(const std::string& path) = 0;
    virtual bool CheckFileExists(const std::string& path) = 0;
    virtual std::shared_ptr<IBlob> ReadFile(const std::string& path) = 0;
    virtual bool WriteFile(const std::string& path, const void* data, size_t size) = 0;
    virtual int32_t EnumerateFiles(std::string& path, const std::vector<std::string>& extensions, EnumerateCallBack callback, bool allow_duplicates = false) = 0;
    virtual int32_t EnumerateDirectories(std::string& path, EnumerateCallBack call_back, bool allow_duplicates = false) = 0;
};

/**
* @brief ネイティブのファイルシステム
*/
class DefaultFileSystem : public IFileSystem {
public:
    /**
    * @brief ディレクトリが存在するかを判定
    * @param[in] path 判定するパス
    * @return 存在するかどうか
    */
    bool CheckDirectoryExists(const std::string& path) override;

    /**
    * @brief ファイルが存在するかを判定
    * @param[in] path 判定するパス
    * @return 存在するかどうか
    */
    bool CheckFileExists(const std::string& path) override;

    /**
    * @brief ファイルへ書き込み
    * @param[in] path 書き込み先のパス名
    * @param[in] extensions 拡張子の配列
    * @param[in] call_bacl
    * @param[in] allow_duplicate
    * @return 成功したか
    */
    std::shared_ptr<IBlob> ReadFile(const std::string& path) override;

    /**
    * @brief ファイルへ書き込み
    * @param[in] path 書き込み先のパス名
    * @param[in] data データのアドレス
    * @param[in] size データのサイズ
    * @return 成功したか
    */
    bool WriteFile(const std::string& path, const void* data, size_t size) override;

    /**
    * @brief ファイル名を列挙
    * @param[in] path 書き込み先のパス名
    * @param[in] extensions 検索対象の拡張子
    * @param[in] callback 発見時に呼ばれるコールバック
    * @param[in] allow_duplicate 複製を許容するか
    * @return 列挙数
    */
    int32_t EnumerateFiles(std::string& path, const std::vector<std::string>& extensions, EnumerateCallBack callback, bool allow_duplicates = false) override;

    /**
    * @brief ディレクトリ名を列挙
    * @param[in] path 書き込み先のパス名
    * @param[in] callback 発見時によばれるコールバック
    * @param[in] allow_duplicate 複製を許容するか
    * @return 列挙数
    */
    int32_t EnumerateDirectories(std::string& path, EnumerateCallBack call_back, bool allow_duplicates = false) override;
};
} // namespace slug