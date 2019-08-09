# What is pruning? Why is it important?

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.40)

# DecisionTreeClassifier offers various parameters to prune your tree, such as:
#   max_depth: You can limit the depth of the tree to make it a generalized model.
#   max_leaf_nodes: Limit the number of leaf nodes.
#   min_samples_leaf: Minimum samples in a leaf node.

# Let's use different options and see how our accuracy varies
# Model 1
model = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5, min_samples_leaf=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Model 2
model = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10, min_samples_leaf=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
