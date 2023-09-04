using Newtonsoft.Json.Linq;
using System.Diagnostics;
using System.Text;

namespace ToolExecuter
{
    class Argument
    {
        public void PerseArgument(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-i" || args[i] == "-r")
                {
                    string input_name = args[i + 1];
                    int begin = input_name.LastIndexOf(".");
                    prefix = input_name.Substring(begin);
                }
            }
        }
        public string prefix = "";
    }

    class Program
    {
        static void SearchRootDirectory(DirectoryInfo di, ref String root_path)
        {
            var file_info_list = di.GetFiles();
            bool is_found = false;
            foreach (var file_info in file_info_list)
            {
                if (file_info.Name == ".root")
                {
                    is_found = true;
                    break;
                }
            }
            if (is_found)
            {
                root_path = di.FullName + "\\";
            }
            else if (di.Parent != null)
            {
                SearchRootDirectory(di.Parent, ref root_path);
            }
        }
        static void Main(string[] args)
        {
            // コマンドライン引き数チェック
            if (args.Length <= 0)
            {
                System.Console.WriteLine("引数を入力してください");
                return;
            }

            // ルートパスを取得
            string root_path = "";
            {
                DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory);
                SearchRootDirectory(di, ref root_path);
            }

            // コマンドライン引き数から必要な情報をパース
            Argument param = new Argument();
            param.PerseArgument(args);

            //Configファイルを読み込む
            string config_path = root_path + "\\tool\\tool_executer\\bin\\ExecuteConfig.json";
            StreamReader sr = new StreamReader(config_path, Encoding.UTF8);
            string config_data = sr.ReadToEnd();
            sr.Close();

            // JsonObjectにする。
            var j_object = JObject.Parse(config_data);
            if (j_object == null)
            {
                return;
            }

            // 変換データリストを抽出
            var convert_data_list = (JObject)j_object["ConvertDataList"];
            if (convert_data_list == null)
            {
                return;
            }

            // コマンドライン引き数と見比べて合致する項目があれば関連ツールの起動コマンドを作成する。
            string execute_file = "";
            foreach (var convert_data in convert_data_list)
            {
                if (convert_data.Key == param.prefix)
                {
                    execute_file = root_path + convert_data.Value.ToString();
                    break;
                }
            }

            // コマンド実行先があるか確認
            if (!File.Exists(execute_file))
            {
                return;
            }

            // コマンドに元のコマンドライン引き数を結合する。
            string argument_list = "";
            foreach (var arg in args)
            {
                argument_list += arg;
                argument_list += " ";
            }

            // コマンド実行
            Process.Start(execute_file, argument_list);

        }
    }
}