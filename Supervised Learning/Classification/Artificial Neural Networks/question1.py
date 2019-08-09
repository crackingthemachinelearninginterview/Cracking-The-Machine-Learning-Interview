# What is an Artificial Neural Network?

# Scikit-Learn has neural_network module which provides Multi-Layer Perceptron Classifier and
# Multi-Layer Perceptron Regressor classes.
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)

# Below code would create a Multi-Layer Perceptron classifier with single hidden layer having 6 hidden units.
mlp_classifier = MLPClassifier(hidden_layer_sizes=6)
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
# Let's look at the metrics of this model
print(classification_report(y_test, y_pred))

# You can create multiple hidden layers as:
mlp_classifier_multi_hidden_layers = MLPClassifier(hidden_layer_sizes=(6, 4, 8))
mlp_classifier_multi_hidden_layers.fit(X_train, y_train)
y_pred = mlp_classifier_multi_hidden_layers.predict(X_test)
# Let's see how the metrics change with change in the number of hidden layers
print(classification_report(y_test, y_pred))

# MLPClassifier also offers parameters such as `activation`,
# `batch_size`, `learning_rate`, 'learning_rate_init' etc.
# Try and play with it by setting different values of these parameters and check how it affects various metrics.
