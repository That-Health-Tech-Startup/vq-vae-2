import torch 
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load and rename columns. Convert stored tensor into list
mnist_data = pd.read_csv('MNISTLatent.csv').rename(columns = {"latent space representation": "latent_space", "labels": "label"})
mnist_data["latent_space"] = mnist_data["latent_space"].map(lambda x: x.replace("\r\n", "")[7:-1]).map(lambda x: json.loads(x)).map(lambda x: [item for sub in x for item in sub])
mnist_data["label"] = mnist_data["label"].map(lambda x: str(x))
# create columns for every latent factor
for i in range(len(mnist_data.latent_space[0])):
    mnist_data["latent_" + str(i)] = mnist_data.latent_space.map(lambda x: x[i])
    
# pca
features = mnist_data.iloc[:, 3:].values
standard_features = StandardScaler().fit_transform(features)

pca = PCA(n_components = 2)
princ_comp = pd.DataFrame(pca.fit_transform(standard_features), columns = ['pca_1', 'pca_2'])
princ_comp['label'] = mnist_data.label

# plot components
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:brown', 'tab:gray']
for target, color in zip(targets, colors):
    indicesToKeep = princ_comp['label'] == target
    ax.scatter(princ_comp.loc[indicesToKeep, 'pca_1']
               , princ_comp.loc[indicesToKeep, 'pca_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

print(pca.explained_variance_ratio_)