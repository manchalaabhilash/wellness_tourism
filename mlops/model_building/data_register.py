from huggingface_hub import HfApi, create_repo, upload_file
import os
from google.colab import userdata

# 🔑 Get token from Colab secrets
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
hf_token = os.environ["HF_TOKEN"]  # Make sure you've added it in Colab: Runtime → Secrets

# Repo details
repo_id = "abhilashmanchala/tourism_data"  # Your dataset repo
repo_type = "dataset"

# Init API with token
api = HfApi(token=hf_token)

# ✅ Create repo if not exists
create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=hf_token)

# 📂 Path to your CSV in Colab runtime
local_csv_path = "mlops/data/tourism.csv"  # change if your file is elsewhere

# ⬆ Upload file
upload_file(
    path_or_fileobj=local_csv_path,
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type,
    token=hf_token
)

print(f"✅ Uploaded {local_csv_path} to https://huggingface.co/datasets/{repo_id}")
