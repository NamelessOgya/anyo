"""
    python3 -m src.third_party.download_llama.download_model
"""

from huggingface_hub import login, snapshot_download



# ★ここを欲しいモデルIDに変える
#model_id = "meta-llama/Llama-3.2-3B"
model_id = "meta-llama/Llama-2-7B-hf"

# ★保存先（好きなパスでOK）
#local_dir = "./models/llama32-3b"
local_dir = "./models/llama2-7b-hf"

login()


if __name__ == "__main__":
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 実体コピーしたいので False 推奨
    )

    print("✅ Downloaded", model_id, "to", local_dir)
