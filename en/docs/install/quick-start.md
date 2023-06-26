# クイックスタート

ここでは、EIPLの環境が適切にインストールされたかを確認するために、学習済み重みと[空間注意機構付き動作生成モデル（SARNN：Spatial Attention with Recurrent Neural Network）](../model/SARNN.md)を用いて検証を行う。
モデルの具体的な学習方法やモデルの詳細については次章以降を参照ください。

## 推論
以下に、学習済みの重みとを用いたSARNNの推論方法を示す。
チュートリアルフォルダ内の`test.py`を実行すると推論結果が`output`フォルダに保存される。
このとき、`--pretrained`引数を指定することで、学習済み重みとサンプルデータが自動的にダウンロードされる。

``` bash linenums="1"
$ cd eipl/tutorials/sarnn
$ python3 ./bin/test.py --pretrained
$ ls ./output/
SARNN_20230514_2312_17_4_1.0.gif
```

## 結果
下図は推論の結果を示しており、図中内の青点は画像から抽出した注意点、そして赤点はRNNが予測した注意点であり、ロボットハンドと把持対象物に着目しながら関節角度を予測していることがわかる。

![SARNNを用いた推論結果](img/sarnn-rt_4.webp){: .center}


## ヘルプ
プログラムが適切に実行できない場合、以下3つの原因が考えられる。

1. **インストールエラー**
    ライブラリが適切にインストールされていない可能性があるため、
    `pip freeze`コマンドを用いて、インストールされているか確認してください。
    もしライブラリがインストールされている場合、そのバージョン情報が表示されます。
    表示されなければ、パッケージがインストールされていない可能性があるため、再度[インストール手順](./install-software.md#pip_install)を確認してください。

        pip freeze | grep eipl


2. **ダウンロードエラー**

    Proxyなどが原因でサンプルデータや学習済みモデルを用いた推論ができない場合は、[重みファイル](https://dl.dropboxusercontent.com/s/o29j0kiqwtqlk9v/pretrained.tar)と[データセット](https://dl.dropboxusercontent.com/s/5gz1j4uzpzhnttt/grasp_bottle.tar)を手動でダウンロードし、`~/.eipl/` フォルダ内に保存した後に、展開してください。
        
        $ cd ~/
        $ mkdir .eipl
        $ cd .eipl
        $ # copy grasp_bottle.tar and pretrained.tar to ~/.eipl/ directory
        $ tar xvf grasp_bottle.tar && tar xvf pretrained.tar
        $ ls grasp_bottle/*
        grasp_bottle/joint_bounds.npy
        ...
        $ ls pretrained/*
        pretrained/CAEBN:
        args.json  model.pth
        ...


3. **描画エラー**

    プログラム実行後に下記のようなエラーが表示される場合、アニメーションファイルの生成に失敗している可能性がある。
    この場合、アニメーション描画時のwriterを変更することで解決する。
    
        File "/usr/lib/python3/dist-packages/matplotlib/animation.py", line 410, in cleanup
            raise subprocess.CalledProcessError(
        subprocess.CalledProcessError: Command '['ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '720x300', '-pix_fmt', 'rgba', '-r', '52.63157894736842', '-loglevel', 'error', '-i', 'pipe:', '-vcodec', 'h264', '-pix_fmt', 'yuv420p', '-y', './output/CAE-RNN-RT_20230510_0134_03_0_0.8.gif']' returned non-zero exit status 1.


    初めに、aptを用いてimagemagickとffmpegをインストールする。
        
        $ sudo apt install imagemagick
        $ sudo apt install ffmpeg
    
    次に、`test.py`の最下部のコードを以下のように編集することでwriterを指定することが可能である。

        # imagemagickを用いる場合
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="imagemagick") 
        
        # ffmpegを用いる場合
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="ffmpeg") 
