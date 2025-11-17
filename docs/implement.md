# ANYO LLM→Rec 蒸留基盤 実装メモ

## 1. 現在までの実装状況
- core/ : ログ・メトリクス・time計測まで実装済み
- teacher/ilora_model.py : LoRAエキスパート定義まで、ゲート合成はTODO
- student/models.py : SASRec 実装済み, Movielensで簡易テスト済み
- distill/kd_losses.py : ランキング蒸留のみ実装, embedding蒸留は未完 など

## 2. コンテナ・環境
- Python: 3.x
- イメージ: anyo-ilora-dllm2rec:latest（Dockerfileからビルド）
- 起動: `env_shell/start_experiment_container.sh`

## 3. 実行方法メモ
- 教師学習: `poetry run python -m src.exp.run_teacher dataset=movielens`
- 生徒ベースライン: `poetry run python -m src.exp.run_student_baseline dataset=movielens`
- 蒸留: `poetry run python -m src.exp.run_distill dataset=movielens`

## 4. 既知の問題 / TODO
- 例: 「iLoRAのプロンプト設計は暫定、要改善」
- 例: 「Amazon Games データセットは未検証」

## 5. 次の実装者へのコメント
- 例: 「まず run_teacher を動かして teacher_outputs の形式を確認すると理解しやすいです」
- 例: 「DLLM2Rec のリポジトリの data_reader 周りを読むと data_bridge の設計に役立ちます」