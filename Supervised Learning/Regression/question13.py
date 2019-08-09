# When would you use k-Nearest Neighbors for regression?

from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

# Let's create a small random dataset
np.random.seed(0)
X = 15 * np.random.rand(50, 1)
y = 5 * np.random.rand(50, 1)
X_test = [[1], [3], [5], [7], [9], [11], [13]]

# We will use k=5 in our example and try different weights for our model.
n_neighbors = 5
weights = ['uniform', 'distance']
for i, weight in enumerate(weights):
    knn_regressor = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weight)
    knn_regressor.fit(X, y)
    y_pred = knn_regressor.predict(X_test)
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='r', label='data')
    plt.scatter(X_test, y_pred, c='g', label='prediction')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights[i]))

plt.show()
