# 模倣学習AI＋固定ジョイントによる配送ロボット — 実行版 v1

## 何が学習済みで、何が学習ではないか
- Pick地点までの移動：学習済みBehavior Cloning（既存 `bc_pick_policy.npz`）
- Pick後の後退・左旋回・配送地点までの移動：学習済みBehavior Cloning（同梱 `bc_delivery_policy.npz`）
- PickとPlaceのアーム関節制御：既存の記録軌道（AIではない）
- 箱の保持：Gazebo Fortress DetachableJoint（AIではない）
- 経路のフェーズ切替・誤差確認・停止：ルールベース
- 強化学習ではありません。

## インストール
Macターミナル：

```bash
docker cp ~/Downloads/bc_full_fixed_delivery_v1.zip b1cabfcea4f4:/tmp/
```

Ubuntuコンテナ：

```bash
mkdir -p /workspace/mobile_crane/course_v1
python3 -m zipfile -e /tmp/bc_full_fixed_delivery_v1.zip /workspace/mobile_crane/course_v1
source ~/.bashrc
python3 /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py check
```

## まず最短の「現在の固定済み箱をAIで最後まで運ぶ」実験

今回すでに、箱を固定したロボットが後退して X ≈ -0.222m にいる。**このままRESETせず**、
以下を試す。`resume`は**箱を固定したまま**配送用BCモデルで残りの後退、90°旋回、配送先まで前進、既存Place、固定解除・開閉まで実行する。

```bash
python3 /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py resume
```

途中でエラーになったら自動停止し、環境は原則そのまま残す。強引にRESETしないこと。

## その後の全自動テスト（**未検証**）

```bash
python3 /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py full
```

実装：古い固定ジョイントの解除 → ソフトRESET → Pick用AIで走行 → 記録したPick → 接続 → 配送用AIで走行 → 記録したPlace → 切断 → グリッパ開放。

**注意**：Gazebo Fortressでは子モデルが親モデルと接触している場合に再接続が失敗することがある。ソフトRESETも物理・速度状態を完全には初期化しない。このため `full` は最初から成功するとは保証しない。失敗したらGUI、serverログ、標準出力を見て原因を切り分ける。必要ならGazeboを新規起動し、新しいジョイントをPick後に追加する方が確実。

## 成功条件と制約
- 配送用AIの走行は座標取得とニューラルネットワーク推論の閉ループ制御。
- 運搬中は箱と車体の相対位置、箱の高さ、コース境界を検査。
- 配送時の箱中心と台中心のXY誤差を検証してから固定解除。
- 動的プラグインを使うため、既存のGazebo Serverとブリッジ、3つのactiveコントローラが必要。
- このスクリプトをGazeboで動作検証したわけではありません。実験時はGUIを観察し、異常ならCtrl+C。
- ROS2 bridge、コントローラ、physicsの挙動によっては停止やエラーとなり得ます。
