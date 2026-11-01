Behavior Cloning 走行AI / 移動式 CRANE+ V2
=====================================

目的：Gazeboの車体をAIがSTART(-0.55,0,0)からPick停止地点(0,0,0)へ運転。
把持・配送・箱の保持はこのスクリプトには含まれません。

技術上の注意:
・これは模倣学習です。2D環境の幾何学的フィードバック教師による操縦デモを生成し、その状態→速度指令をNumPy製MLPに学習させています。
・以前収集した54行のGazebo CSVは、この版では学習に使用していません。少数かつ状態と行動の時刻が不明確なためです。
・60/60の成功は2D簡易シミュレーションでの評価で、Gazeboでの成功は未確認です。
・学習済みネットワークはbc_pick_policy.npz。Gazeboモードで使用するのはニューラルネットワークの出力で、教師の制御式ではありません。

インストール：Mac側でZIPをダウンロードしてDockerへコピー
  docker cp ~/Downloads/bc_pick_navigation_v1.zip b1cabfcea4f4:/tmp/

Ubuntu側（別ターミナルでGazebo、robot_state_publisher、controller、clock/cmd_velブリッジを起動済みのこと）:
  mkdir -p /workspace/mobile_crane/course_v1
  python3 -m zipfile -e /tmp/bc_pick_navigation_v1.zip /workspace/mobile_crane/course_v1
  source ~/.bashrc
  python3 -c 'import numpy; print(numpy.__version__)'
  python3 /workspace/mobile_crane/course_v1/bc_pick_navigation.py eval

学習を自分のDockerでも再実行する場合:
  OPENBLAS_NUM_THREADS=1 python3 /workspace/mobile_crane/course_v1/bc_pick_navigation.py train

実際のGazeboでAIを走行させる場合:
  python3 /workspace/mobile_crane/course_v1/reset_course.py --apply
  python3 /workspace/mobile_crane/course_v1/bc_pick_navigation.py gazebo

重要:
・自動配送プログラム（run_auto_delivery.py）やWASDを同時に実行しないこと。
・箱を掴む操作は行いません。AIがPick停止位置まで到達したか確認する試験です。
・走行中の異常を確認したら、Gazeboを一時停止。停止位置チェックと最終停止指令が組み込まれていますが、衝突回避は保証しません。
