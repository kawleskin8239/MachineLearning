import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering
df = pd.read_csv("customer_segmentation.csv")

# Sum previous campaign acceptance
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df['total_campaigns'] = df[campaign_columns].sum(axis=1)

middle_columns = list(df.columns[8:-10]) + ['Complain', 'total_campaigns'] # Get columns X will have

df.dropna(inplace=True)

X = df[middle_columns].copy()

# Normalizes the number columns
mean = np.mean(X, axis=0)
X -= mean
std = np.std(X, axis=0).to_numpy()
X /= std

X = X.to_numpy()


inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(X)
    inertia[i-1] = kmeans.inertia_

plt.plot(np.arange(1, 15), inertia)
plt.show()
