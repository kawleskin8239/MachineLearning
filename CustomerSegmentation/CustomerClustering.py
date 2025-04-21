import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

#https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering
df = pd.read_csv("customer_segmentation.csv")

# Sums accepted campaigns into one column
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df['total_campaigns'] = df[campaign_columns].sum(axis=1)

# Creates a list of all the features being used for clustering
middle_columns = list(df.columns[8:-10]) + ['Complain', 'total_campaigns'] # Get columns X will have

df.dropna(inplace=True)

X = df[middle_columns].copy()

# Normalize data
mean = np.mean(X, axis=0)
X -= mean
std = np.std(X, axis=0).to_numpy()
X /= std

# Make X a numpy array
X = X.to_numpy()

kmeans = KMeans(n_clusters=8) # Based on Elbow.py optimal # of clusters is 6-8
kmeans.fit(X)

# Undo the normalization of data
real_centers = pd.DataFrame(kmeans.cluster_centers_, columns=middle_columns)
real_centers = (real_centers * std) + mean

pd.set_option('display.max_columns', None)  # Displays all columns
pd.set_option('display.float_format', '{:.6f}'.format) # Prevents scientific notation

print(kmeans.inertia_)
print("Average stats for clusters")
print(real_centers)
