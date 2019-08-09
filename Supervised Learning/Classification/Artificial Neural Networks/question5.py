# What is an activation function?

from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)

# MLPClassifier offers a parameter `activation` which can take different values such as
# ‘identity’, ‘logistic’, ‘tanh’, ‘relu’. Default value is `relu`.
# You can set it as follows:
mlp_classifier = MLPClassifier(hidden_layer_sizes=6, activation='identity')
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
# Let's look at the metrics of this model
print(classification_report(y_test, y_pred))

mlp_classifier = MLPClassifier(hidden_layer_sizes=6, activation='tanh')
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
# Let's see how the metrics change with change in the activation type.
print(classification_report(y_test, y_pred))
