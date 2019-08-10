# Why and where do you need to use Cluster analysis?

# Scikit-Learn provides `cluster` module to use clustering algorithms.
# In this example, we will use simple KMeans Clustering model.

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Let's create our sample dataset for training
data_type = np.zeros((60, 2))
data_type[0:20, 0] = np.random.uniform(-3.5, -1.5, size=20)
data_type[0:20, 1] = np.random.uniform(-1, 0, size=20)

data_type[20:40, 0] = np.random.uniform(0, 4, size=20)
data_type[20:40, 1] = np.random.uniform(-3, -1, size=20)

data_type[40:60, 0] = np.random.uniform(0, 3, size=20)
data_type[40:60, 1] = np.random.uniform(0, 2, size=20)

# We have number of clusters=3
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(data_type)

# Let's plot our training dataset with cluster labels assigned by our KMeans model
colors = ['red', 'green', 'blue']
for i in range(len(data_type)):
    # let's get the color from model's label
    cluster_label = kmeans_model.labels_[i]

    # Now, plot the point with corresponding color for the label
    plt.scatter(data_type[i][0], data_type[i][1], color=colors[cluster_label])

plt.show()
