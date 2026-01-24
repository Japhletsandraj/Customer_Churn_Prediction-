import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
os.environ["KAGGLEHUB_CACHE"] = "./dataset"
path = kagglehub.dataset_download("abdullah0a/telecom-customer-churn-insights-for-analysis")
print(f"Data saved at: {path}")