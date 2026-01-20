import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#dataset
dataset_path = "dataset/customer_churn_dataset.csv"

# Load dataset
data = pd.read_csv(dataset_path)
print(data.head())

#visualize the number of churned vs non-churned customers
sns.countplot(x='Churn',data=data, palette='Set1',legend=False)
plt.title("Churn distribution")
plt.ylabel("number of customers")
plt.xlabel("churn")
plt.show()

#data preprocessing
data = pd.to_numeric(data, errors='coerce')
data.fillna(data.mean(), inplace=True)

# splitting into test and train
x = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)