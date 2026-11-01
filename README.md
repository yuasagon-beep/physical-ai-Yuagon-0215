# 模倣学習による自律配送ロボット｜Mobile CRANE+ V2

ROS 2とIgnition Gazebo Fortressを使用し、移動台車に搭載したCRANE+ V2アームで箱を受け取り、配送台まで運んで置くシミュレーションです。**移動台車の走行制御には、人間の操作データから学習したBehavior Cloning（BC：模倣学習）を使用**しています。

> **デモ動画（YouTube）**：公開済みの動画URLをここに記載してください。

## 実現した動作

1. ロボットと箱を初期位置にリセット
2. **学習済みBCモデル**で箱の前まで前進
3. **記録済みのアーム・グリッパー軌道**を再生して箱を把持（Pick）
4. Gazeboの**DetachableJoint**プラグインで箱とグリッパーを固定
5. **学習済みBCモデル**で箱を保持したまま後退
6. 約90°旋回して配送方向へ向きを変える
7. **学習済みBCモデル**で配送台へ前進
8. **記録済み軌道**で箱を配置（Place）し、固定解除・グリッパー開放
9. 箱の最終位置を確認して配送完了を判定

## 使用技術と学習の範囲

| 対象 | 手法 |
| --- | --- |
| 移動台車の前進・後退・旋回 | 人間の走行デモから学習したBehavior Cloning（模倣学習） |
| アームのPick / Place | あらかじめ記録した関節軌道の再生 |
| 箱の運搬中の保持 | Gazebo `DetachableJoint` によるシミュレーション上の固定 |
| 実行順序・位置チェック | Pythonによる自動実行と安全判定 |

**注意：** 学習済みAIが制御しているのは移動台車の走行部分です。アームのPick/Placeは学習制御ではなく記録軌道の再生であり、箱の保持にはシミュレーション補助機能を使用しています。強化学習（RL）は使用していません。

## 動作環境

- ROS 2 Humble
- Ignition Gazebo Fortress（Gazebo 6）
- CRANE+ V2のロボットモデルと`ros2_control`
- Python 3 / NumPy / ROS 2 Pythonパッケージ
- 開発時はDocker内で実行

CRANE+ V2の説明ファイル・メッシュ・コントローラー関連パッケージなど、リポジトリ外の依存関係が必要な場合があります。これらは配布元とライセンスを確認のうえ別途導入してください。また、一部のSDF・設定ファイルには開発環境固有の絶対パスが含まれるため、別のPCで動かす場合は調整が必要です。

## 主なファイル

| ファイル | 内容 |
| --- | --- |
| `course_v1/bc_full_fixed_delivery.py` | STARTから配送完了までの統合実行（`check`・`full`・再開モード） |
| `course_v1/bc_pick_navigation.py` | Pick地点までのBC走行 |
| `course_v1/bc_delivery_navigation.py` | 後退・旋回・配送地点までのBC走行 |
| `course_v1/bc_pick_policy.npz` | Pick地点への走行用学習済みモデル |
| `course_v1/bc_delivery_policy.npz` | 配送用走行の学習済みモデル |
| `course_v1/reset_course.py` | ロボットと箱の初期位置へのソフトリセット |
| `course_v1/course_world.sdf` | 配送コースのGazeboワールド |
| `mobile_crane.absolute_mesh.sdf` | 移動式CRANE+ V2モデル（旋回改善のためのキャスター摩擦設定を含む） |
| `replay_trajectory.py` | Pick用の記録済み軌道の再生 |
| `place_arm_full_test.py` | Place用の記録済み軌道の再生 |
| `recordings/` | 記録済み関節軌道など |

## 実行方法

ROS 2・Gazebo・ロボットモデル・ROS-Gazebo Bridgeを起動し、`joint_state_broadcaster`、`crane_plus_arm_controller`、`crane_plus_gripper_controller`が`active`になっていることを確認してください。

以下は**開発時のDocker環境内**での実行例です。

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash

# 環境・ロボットの状態確認
python3 /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py check

# 初期化から配送完了まで一括実行
python3 -u /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py full
```

`full`は実行前にコースのソフトリセットを行います。箱が固定されたまま異常停止した場合は、**固定ジョイントの状態を確認するまで再リセットしないでください**。負荷が高い環境ではGazeboの位置取得がタイムアウトする場合があります。

## 実験結果・再現性について

構築したシミュレーション環境で、箱の把持から後退・90°旋回・配送地点への移動・箱の解放までを一連の動作として実行できました。別環境での完全な再現性は未検証です。ROS 2 / Gazeboのバージョン、プラグイン、モデルのメッシュや絶対パス、コントローラー設定などを確認してください。
