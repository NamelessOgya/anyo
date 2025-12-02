# iLoRA機能廃止に伴う不要ファイル・処理の精査

今後iLoRA機能を廃止し、teacher機能は `bigrec` および `MOE-bigrec` のみに集約するため、以下のファイルおよび処理が不要となります。

## 1. 不要となるファイル (削除対象)

以下のファイルは iLoRA 機能に特化しており、他の機能では使用されていないため、削除可能です。

### ソースコード (`src/`)
- `src/teacher/ilora_model.py`: iLoRAモデルの主要実装クラス。
- `src/teacher/trainer_ilora.py`: iLoRA専用のTrainerクラス（蒸留ロジックなどを含む）。
- `src/teacher/interfaces.py`: `TeacherModel` インターフェース定義。`BigRecModel` や `MoEBigRecModel` は `pl.LightningModule` を直接継承しており、このインターフェースを使用していません。
- `reproduce_ilora_bug.py`: iLoRAのバグ再現用スクリプト。

### 設定ファイル (`conf/`)
- `conf/teacher/ilora.yaml`: iLoRAモデルのデフォルト設定。
- `conf/experiment/ilora_distill.yaml`: iLoRA蒸留実験用設定。
- `conf/experiment/ilora_movielens.yaml`: iLoRA MovieLens実験用設定。
- `conf/experiment/ilora_student.yaml`: iLoRA生徒モデル実験用設定。
- `conf/experiment/ilora_teacher.yaml`: iLoRA教師モデル実験用設定。

### テストコード (`tests/`)
- `tests/cpu/teacher/test_ilora_model.py`
- `tests/cpu/teacher/test_ilora_optimizations.py`
- `tests/cpu/teacher/test_trainer_ilora.py`
- `tests/cpu/reproducibility/test_ilora_reproducibility.py`
- `tests/teacher/test_ilora_trainer_e4srec.py`

## 2. 不要となる処理 (修正対象)

以下のファイルには iLoRA とその他のモデル (`bigrec`, `moe_bigrec`) のロジックが混在しています。iLoRA 関連の分岐や処理を削除する必要があります。

### `src/teacher/factory.py`
- **該当箇所**: `create_teacher_model` 関数内の `if model_type == "ilora":` ブロック全体。
- **詳細**: iLoRAモデルの初期化、LLMのロード（iLoRA特有のフック設定など）、パラメータ凍結ロジックが含まれています。これらは削除可能です。

### `src/exp/run_teacher.py`
- **該当箇所**:
    - `model_type` 判定による分岐ロジック。
    - `else: # iLoRA Logic` ブロック（`iLoRATrainer` のインスタンス化など）。
    - `EarlyStopping` コールバックの設定（`val_hr@10` を監視する iLoRA 特有の設定）。
    - テスト実行部分の `if model_type == "ilora":` ブロック。
- **詳細**: `bigrec` と `moe_bigrec` は `pl.Trainer.fit` を使用し、`BigRecCollator` を使用するフローになっていますが、iLoRA は独自の `iLoRATrainer` を使用しています。この iLoRA 用のフローを削除し、コードを簡素化できます。

## 3. 役割が重複しているファイル・処理

現状、複数のモデルタイプをサポートするために、似たような役割を持つファイルや処理が存在しています。iLoRA 廃止により、これらは解消または整理されます。

| 役割 | iLoRA (廃止) | BigRec / MoE-BigRec (存続) | 重複・整理の方向性 |
| :--- | :--- | :--- | :--- |
| **モデル定義** | `src/teacher/ilora_model.py` | `src/teacher/bigrec_model.py`<br>`src/teacher/moe_bigrec_model.py` | `ilora_model.py` を削除することで重複解消。 |
| **学習ループ** | `src/teacher/trainer_ilora.py` (カスタムTrainer) | `src/exp/run_teacher.py` 内の `pl.Trainer` + `BigRecCollator` | iLoRA は独自のTrainerクラスを持っていたが、BigRec系はLightning標準のTrainerを使用。`trainer_ilora.py` を削除し、`run_teacher.py` をLightning標準フローに統一。 |
| **インターフェース** | `src/teacher/interfaces.py` (`TeacherModel`) | なし (`pl.LightningModule` を直接継承) | `TeacherModel` インターフェースは iLoRA でのみ使用されていたため、ファイルごと削除。 |
| **初期化ファクトリ** | `src/teacher/factory.py` | `src/teacher/factory.py` | 同一ファイル内で分岐していたが、iLoRA 用の分岐を削除することでシンプルになる。 |
| **実行スクリプト** | `src/exp/run_teacher.py` | `src/exp/run_teacher.py` | 同一ファイル内で分岐していたが、iLoRA 用の分岐を削除することでシンプルになる。 |

## まとめ

iLoRA 機能の廃止により、専用のモデル定義、Trainer、設定ファイル、テストコードを削除できます。また、`factory.py` や `run_teacher.py` などの共通スクリプトから iLoRA 用の分岐ロジックを取り除くことで、コードベースの見通しが良くなり、メンテナンス性が向上します。
