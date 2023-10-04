# 概要

深層学習フレームワークとしてはPytorchを採用しており、最新版の [pytorch](https://pytorch.org/get-started/locally/) をインストールすることを推奨する。
特に、Pytorch 2.0以上は、事前にコンパイルされたことにより、学習速度を向上させ、GPUメモリの使用率を低減させるため、大規模なモデルの学習を高速に実行可能である。
なお、CUDAやNvidiaドライバーは利用するPytorchの[バージョン](https://pytorch.org/get-started/previous-versions/)に合わせてインストールする必要がある。

----
## ファイル
本ライブラリは以下のように構成される。

- **data**: サンプルデータのダウンロードや、モデル学習のためのDataloader
- **layer**: [階層型RNN](../zoo/MTRNN.md)や[空間注意機構](../model/SARNN.md#spatial_softmax)などを実装
- **model**: 複数の動作生成モデルを実装、入力は関節角度（任意の自由度）とカラー画像（128x128ピクセル）
- **test**: テストプログラム
- **utils**: 正規化や可視化、引数処理などの関数

----
## pip {#pip_install}

GithubからEIPLのリポジトリクローンし、pipコマンドを用いて環境をインストールする。

```bash linenums="1"
mkdir ~/work/
cd ~/work/
git clone https://github.com/ogata-lab/eipl.git
cd eipl
pip install -r requirements.txt
pip install -e .
```
