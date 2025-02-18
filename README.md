# LLMを方策とする会話に基づくマルチエージェント強化学習
## 概要

2024年度卒 浅間 慶二郎

表題の研究にてシミュレーション実験で用いたコード一式です。

## 環境構築
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

## ディレクトリ構成
.<br>
├── README.md<br>
├── babyai # BabyAI環境をマルチエージェントに改変したもの<br>
├── config # 実験設定ファイル<br>
├── gym_minigrid # gym_minigridライブラリをマルチエージェントに改変したもの<br>
├── logger # ログの保存などの処理<br>
├── result # 結果を保存する場所<br>
├── utils # 学習や環境に関する細かい処理のコード<br>
├── arial.ttf # 結果出力に使うフォント(上手いことやればなくてもできるはず)<br>
├── ENV.py # OpenAIのキーなどを書くところ ignore推奨<br> 
├── executed_configs.py # 実行するコンフィグファイル一覧<br>
├── gpu_checker.py # 研究室内のGPUメモリ争奪戦で勝利するためのコード<br>
├── main_restart.py # mainで実行した実験を途中から再開するコード<br>
├── main.py # 実験を実行するコード<br>
├── policy.py # エージェントの方策に関するコード<br>
└── README.md