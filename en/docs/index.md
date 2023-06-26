# はじめに

EIPL（Embodied Intelligence with Deep Predictive Learning）とは、[早稲田大学尾形哲也研究室](https://ogata-lab.jp/ja/)で開発された、深層予測学習を用いたロボット動作生成ライブラリである。
[深層予測学習](overview.md) とは、過去の学習経験に基づいて、実世界に適した動作をリアルタイムに予測することで、未学習の環境や作業対象に対し柔軟な動作を実行可能な手法であり、少ない動作教示コストで動作を学習・生成することが可能である。
ここでは、実ロボットとしてスマートロボット[AIREC（AIREC：AI-driven Robot for Embrace and Care）](https://airec-waseda.jp/)と[Open Manpulator](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/)を対象に、収集した視覚・関節角度データを用いて、モデル実装から学習、リアルタイム動作生成まで体系的に学ぶことが可能である。
また今後、EIPLを用いて新たに開発された動作生成モデルは、[Model Zoo](./zoo/overview.md) に順次公開する予定である。
以下に各章の概要を示す。


1. [**深層予測学習**](overview/)

    深層予測学習のコンセプトとロボット実装に向けた3つのステップ、動作教示、学習、動作生成について述べる。
    
2. [**インストール**](install/install-software)

    EIPLのインストールと、学習済み重みを使った動作確認方法について述べる。

3. [**動作教示**](teach/overview)

    EIPLのサンプルデータとして、AIRECを用いた物体把持動作データセットを提供。<br>タスク概要に加え、ROSbagデータからデータ抽出及びデータセットの作成方法について述べる。

4. [**モデル学習**](model/dataloader)

    注意機構動作生成モデルを例に、モデルの学習から推論までの一連の実装方法について述べる。

5. [**ロボット応用**](robot/overview)

    Open Manpulatorを用いた動作教示から実機動作生成までの一連の手順について述べる。

6. [**ModelZoo**](zoo/overview)

    EIPLを用いて開発された動作生成モデルを順次公開する。

7. [**Tips**](tips/normalization/)

    ロボット動作学習性能を向上させるためのノウハウについて述べる。



----
**謝辞**

本成果は、JST ムーンショット型研究開発事業 JPMJMS2031 の支援を受けたものです。ここに謝意を表します。
