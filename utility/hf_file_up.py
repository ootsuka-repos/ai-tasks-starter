import os
from huggingface_hub import login, HfApi, create_repo
from dotenv import load_dotenv

def upload_folder_to_hf(token, repo_id, repo_type, private, folder_path, path_in_repo):
    login(token=token)
    create_repo(repo_id, repo_type=repo_type, private=private)
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo
    )

if __name__ == "__main__":
    load_dotenv(dotenv_path="./.env")
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID")
    repo_type = os.getenv("HF_REPO_TYPE", "dataset")
    private = os.getenv("HF_REPO_PRIVATE", "true").lower() == "true"
    folder_path = os.getenv("HF_UPLOAD_DIR")
    path_in_repo = os.getenv("HF_PATH_IN_REPO")
    if not all([token, repo_id, repo_type, folder_path, path_in_repo]):
        raise ValueError("必要な環境変数が不足しています")
    upload_folder_to_hf(token, repo_id, repo_type, private, folder_path, path_in_repo)
    print("Upload completed.")