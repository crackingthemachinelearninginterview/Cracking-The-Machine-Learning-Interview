# How should you set the value of the learning rate?

from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import time

iris = datasets.load_iris()

# Let's measure the time taken for our model to converge with a small learning rate
start_time = time.time()
mlp_classifier = MLPClassifier(hidden_layer_sizes=(6,4), learning_rate_init=0.001)
mlp_classifier.fit(iris.data, iris.target)
end_time = time.time()
print("Time taken to converge is " + str(end_time - start_time) + " seconds")

# large learning rate
start_time = time.time()
mlp_classifier = MLPClassifier(hidden_layer_sizes=(6, 4), learning_rate_init=0.1)
mlp_classifier.fit(iris.data, iris.target)
end_time = time.time()
print("Time taken to converge is " + str(end_time - start_time) + " seconds")
