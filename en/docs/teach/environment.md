# 実行環境

<!-- ******************************** -->
----
### ROS
ROSがインストールされている環境でデータセットの作成を行う場合、以下の処理は不要であるため、
[次章](./dataset.md)へ進んでください。


### pyenv
本プログラムでは、`rosbag` からデータを抽出するのに rospy と rosbag パッケージを利用する。
そのためROS（Noetic推奨）をインストール済みのPCであれば、そのまま実行すれば問題なく動作する。

一方で、ROSがインストールされていないPCでrospyなどを使う方法として、
[rospypi/simple](https://github.com/rospypi/simple)が挙げられる。
本パッケージは、ROSをインストールすることなく、rospyやtf2などのバイナリパッケージを利用することが可能である。
さらに、LinuxやWindows、MacOSに対応していることから、収集したデータを自身のPC環境で容易に解析することが可能である。
なお、既存のpython環境との競合を防ぐために、venvを用いて仮想環境を作ることを推奨する。
以下に、venvを用いたrospypi/simpleライブラリの環境構築手順を示す。

```bash
$ python3 -m venv ~/.venv/rosbag
$ source ~/.venv/rosbag/bin/activate
$ pip install -U pip
$ pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
$ pip install matplotlib numpy opencv-python
```

!!! note
    
    rospypi/simpleライブラリですべてのメッセージデータに対応できることは確認できていない。
    特にカスタムROSメッセージは未検証であるため、仮想環境でプログラムが正しく実行できない場合は、
    ROS環境で実行すること。
