# What do you understand by maximal margin classifier? Why is it beneficial?

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# Let's use the standard iris dataset. For simplicity, we will use the 1st two features only.
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Now, let's create a linear SVM classifier instance and fit the iris dataset.
svc_classifier = svm.SVC(kernel='linear').fit(X, y)

# create a mesh to plot in
h = .05  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(2, 2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

predicted_y = svc_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
predicted_y = predicted_y.reshape(xx.shape)
plt.contourf(xx, yy, predicted_y, cmap=plt.cm.coolwarm, alpha=0.6)

# Plot the input training data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
