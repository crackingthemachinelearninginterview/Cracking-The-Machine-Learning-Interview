# What is a Decision Tree?
import matplotlib.pyplot as plt

# Decision Tree Classifier is part of the tree module in Scikit-Learn
from sklearn.tree import DecisionTreeClassifier, plot_tree

#  Let's use the standard iris dataset to train our model.
from sklearn import datasets
[iris_data, iris_target] = datasets.load_iris(return_X_y=True)

dtree = DecisionTreeClassifier()
dtree.fit(iris_data, iris_target)

# tree module provides a plot_tree method to visualize the tree
plt.figure()
plot_tree(decision_tree=dtree)
plt.show()
