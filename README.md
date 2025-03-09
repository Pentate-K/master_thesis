# LLMを方策とする会話に基づくマルチエージェント強化学習
## 概要

2024年度卒 浅間 慶二郎

表題の研究にてシミュレーション実験で用いたコード一式です。

## 環境構築と実行
conda環境を作成
```
conda env create -f environment.yml 
```
conda環境をactivate
```
conda actiavte multi_babyai
```
実験を実行
```
python main.py 
```

※LLMをHuggingFaceから取ってきている場合は実行前にログインしておく
```
huggingface-cli login
```
```
Enter your token (input will not be visible): トークンをペースト
```
モデルによってはHuggingFaceにてアクセスに関する登録をしなければならないので注意

初期設定では"meta-llama/Meta-Llama-3.1-8B-Instruct"を使用

## 実験設定の変更方法
main.pyを実行する際, excuted_configs.py内に記述された設定ファイルに基づいて実行される.

設定ファイルとはconfigフォルダ内のjsonファイルで, 例えば以下はconfig/Debug.jsonの内容である.

```
{
    "extends": [
        "base/General",
        "base/Reflexion",
        "base/PutNext",
        "base/Simple",
        "base/Llama3.1-8B"
    ],

    "hyperparam": {
        "agent_num" : 2,
        "reflexion_type" : "none",
        "is_use_fixed_init_subgoal" : false,
        "trial_count" : 100,
        "free_mode" : false,
        "is_use_consideration" : true,
        "history_size" : 7,
        "env_fixed_seed" : 76
    }
}
```
実験設定が多くなるため, configの継承機能を実装した. "hyperparam"では, 実験に使用するハイパーパラメータの設定を行う. "extends"では, 継承する他のjsonファイルを指定できる. ただし内容が重複した部分は"hyperparam"での設定が優先される.

主なハイパーパラメータは以下の通り.
```
"env_name": 実行する環境名(str)
"max_step": 1エピソードの最大ステップ数(int)
"policy_name": エージェントのポリシー名(str)
"agent_num": エージェント数(int)
"llm_model": LLMのモデル名(str)
"history_size": 履歴の長さ(int)
"trial_count": 実行エピソード数(int)
"reflexion_memory_size": Reflexionで保持する反省文の数(int)
```

## 実行結果
実行結果はデフォルトではresultフォルダに格納される(初回はmain.pyの実行で生成される). 

実行するたびに以下のフォルダに結果やログが格納される.
```
result/実験環境名/ポリシー名/実行日時_config名/
```

## ディレクトリ構成
```
.
├── README.md
├── babyai # BabyAI環境をマルチエージェントに改変したもの
├── config # 実験設定ファイル
├── gym_minigrid # gym_minigridライブラリをマルチエージェントに改変したもの
├── logger # ログの保存などの処理
├── utils # 学習や環境に関する細かい処理のコード
├── arial.ttf # 結果出力に使うフォント(上手いことやればなくてもできるはず)
├── ENV.py # OpenAIのキーなどを書くところ gitignore推奨
├── executed_configs.py # 実行するコンフィグファイル一覧
├── gpu_checker.py # 研究室内のGPUメモリ争奪戦で勝利するためのコード
├── main_restart.py # mainで実行した実験を途中から再開するコード
├── main.py # 実験を実行するコード
├── policy.py # エージェントの方策に関するコード
└── README.md
```