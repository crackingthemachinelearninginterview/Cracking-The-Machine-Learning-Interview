# Tell me about One vs All Logistic Regression.

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

[X, y] = datasets.load_iris(return_X_y=True)

# Create one-vs-rest logistic regression model and train it
clf = LogisticRegression(random_state=0, multi_class='ovr')
clf.fit(X, y)

# Create new observation
new_observation = [[.2, .4, .6, .8]]

# Let's predict its class
print(clf.predict(new_observation))

# The probability of the class predicted should be the highest among all the probabilities.
print(clf.predict_proba(new_observation))
