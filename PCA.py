from sklearn import decomposition
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Download a file from Github
url = "https://raw.githubusercontent.com/AmitSharma8/Principal-component-analysis/main/pca_2d_data.csv"
response = requests.get(url)
output_path = "pca_2d_data.csv"        # local file path
with open(output_path, "wb") as file:  # Write the content to a local file
    file.write(response.content)
    
data = pd.read_csv('pca_2d_data.csv')

# Column standardization
mu = data.mean(axis = 0)
std = data.std(axis = 0)
X_st = (data - mu)/std

pca = decomposition.PCA(n_components=2)

X_p = pca.fit_transform(X_st)

plt.figure()
plt.scatter(X_p[:, 0], np.zeros(20) )
plt.show()
