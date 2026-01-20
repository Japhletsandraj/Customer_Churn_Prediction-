import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
#dataset
dataset_path = "dataset/customer_churn_dataset.csv"

# Load dataset
data = pd.read_csv(dataset_path)
print(data.head())

#visualize the number of churned vs non-churned customers
sns.countplot(x='Churn',data=data, palette='Set1')
plt.title("Churn distribution")
plt.ylabel("number of customers")
plt.xlabel("churn")
plt.show()