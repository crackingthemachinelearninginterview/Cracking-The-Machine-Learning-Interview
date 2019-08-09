# Name some common kernels. How do you select a particular kernel for your problem?

# Let's go back to the code in question 1 and try different kernel methods.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# Let's use the standard iris dataset. For simplicity, we will use the 1st two features only.
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Now, let's create SVM classifier instances and fit the iris dataset.
linear_classifier = svm.SVC(kernel='linear')
linear_classifier.fit(X, y)

poly_classifier = svm.SVC(kernel='poly', degree=3)
poly_classifier.fit(X, y)

rbf_classifier = svm.SVC(kernel='rbf', gamma=0.8)
rbf_classifier.fit(X, y)

# create a mesh to plot in
h = .05  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

titles = ['Linear SVM Classifier', 'Polynomial SVM Classifier', 'RBF SVM Classifier']
for index, classifier in enumerate((linear_classifier, poly_classifier, rbf_classifier)):
    plt.subplot(2, 2, index + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    predicted_y = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    predicted_y = predicted_y.reshape(xx.shape)
    plt.contourf(xx, yy, predicted_y, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(titles[index])

plt.show()
