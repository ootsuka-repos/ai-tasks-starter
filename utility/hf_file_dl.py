import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

def download_model(model_name: str):
    return snapshot_download(model_name)

if __name__ == "__main__":
    load_dotenv(dotenv_path="./.env")
    model_name = os.getenv("HF_MODEL_NAME")
    if not model_name:
        raise ValueError("環境変数HF_MODEL_NAMEが設定されていません")
    model_path = download_model(model_name)
    print(f"Downloaded to: {model_path}")