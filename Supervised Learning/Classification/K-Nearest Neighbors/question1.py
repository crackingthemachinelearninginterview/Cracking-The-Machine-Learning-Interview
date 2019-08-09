# Walk me through the k-Nearest Neighbors algorithm.

# K Nearest Neighbors classifier is present as KNeighborsClassifier class in neighbors module
from sklearn.neighbors import KNeighborsClassifier

# Let's create input dataset discussed in our solution
X = [[150, 45],
     [160, 80],
     [165, 75],
     [170, 70],
     [175, 75]]
Y = ["Normal", "Obese", "Overweight", "Normal", "Normal"]
k = 3

# Let's create the K nearest neighbors classifier with 3 neighbors
classifier = KNeighborsClassifier(n_neighbors=k)

# Now, train our classifier on the input dataset
classifier.fit(X, Y)

# Let's see what it predicts for the person whose height is 180cm and weight is 75kgs.
print(classifier.predict([[180, 75]]))

# As you can see, it also predicts the person type as 'Normal'.
