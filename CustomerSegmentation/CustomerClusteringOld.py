import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

#https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering
df = pd.read_csv("customer_segmentation.csv")

middle_columns = df.columns[1:-3] # Get columns X will have

# gets all float and int columns  in middle_columns to normalize them
num_columns = df[middle_columns].select_dtypes("float").columns.append(df[middle_columns].select_dtypes("int").columns)

# Converts date joined to number of days since they joined from today's date
today = pd.to_datetime('today')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Dt_Customer'] = (today - df['Dt_Customer']).dt.days

# Create a dictionary for category columns
cat_columns = df.select_dtypes("object").columns # Select all columns of tye object (catergory columns)
df[cat_columns] = df[cat_columns].astype("category") # cast them to type category
# Make a dictionary of the different categories converted to numbers
cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
range(len(df[cat_columns[i]].cat.categories))} for i
in range(len(cat_columns))}
print(cat_dict) # Print the dictionary for deciphering the category numbers

# Replace categories with int values
df[df.select_dtypes("category").columns] = df[df.select_dtypes("category").columns].apply(lambda x: x.cat.codes)

df.dropna(inplace=True)

X = df[middle_columns].copy()

# Normalize data
mean = np.mean(X[num_columns], axis=0)
X[num_columns] -= mean
std = np.std(X[num_columns], axis=0).to_numpy()
X[num_columns] /= std

X = X.to_numpy()

kmeans = KMeans(n_clusters=3) # Based on Elbow.py optimal # of clusters is 3
kmeans.fit(X)

real_centers = pd.DataFrame(kmeans.cluster_centers_, columns=middle_columns)

# Undo the normalization of data
real_centers[num_columns] = (real_centers[num_columns] * std) + mean

pd.set_option('display.max_columns', None)  # Displays all columns
pd.set_option('display.float_format', '{:.6f}'.format) # Prevents scientific notation

print(kmeans.inertia_)
print("Average stats for clusters")
print(real_centers)
