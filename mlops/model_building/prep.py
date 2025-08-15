# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
hf_token = os.environ["HF_TOKEN"]  # Make sure you've added it in Colab: Runtime → Secrets
api = HfApi(token=hf_token)

# Define Hugging Face dataset repo
dataset_path = "hf://datasets/abhilashmanchala/tourism_data/tourism.csv"

df = pd.read_csv(dataset_path)
print("loaded successfully")

df.drop(["CustomerID"], axis=1, inplace=True)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="abhilashmanchala/tourism_data",
        repo_type="dataset",
    )
