# LLMを方策とする会話に基づくマルチエージェント強化学習
## 概要

2024年度卒 浅間 慶二郎

表題の研究にてシミュレーション実験で用いたコード一式です。

## 環境構築
conda環境を作成
```
conda env create -f environment.yml 
```
実験を実行
```
python main.py 
```
mainにより実行したい実験はexecuted_configs.pyにより、コンフィグファイル名のリストで指定します。
コンフィグファイルはconfigフォルダ内にあり、自由に追加できます。