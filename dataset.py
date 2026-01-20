import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
os.environ["KAGGLEHUB_CACHE"] = "./dataset"
path = kagglehub.dataset_download("ziya07/customer-churn-prediction-dataset")
print(f"Data saved at: {path}")