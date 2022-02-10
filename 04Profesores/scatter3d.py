'''
==============
3D scatterplot
   Wine PCA
==============
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

df = load_wine(as_frame=True)['data'].apply(zscore)
pca3 = PCA(n_components=3)
pca3.fit(df)
proj_data = pca3.transform(df)

fig = plt.figure()                                                              
ax = fig.add_subplot(111, projection='3d')

ax.scatter(proj_data[:,0], proj_data[:,1], proj_data[:,2], c='r', marker='.')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

plt.show()
