from huggingface_hub import HfApi
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
hf_token = os.environ["HF_TOKEN"]  # Make sure you've added it in Colab: Runtime → Secrets
api = HfApi(token=hf_token)

api.upload_folder(
    folder_path="mlops/deployment",     # the local folder containing your files
    repo_id="abhilashmanchala/wellness_tourism_prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
