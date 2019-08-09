# What are the different stages to learn the hypotheses or models in Machine Learning?

# Le's use Support Vector Machine for our question.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# In this example, we will use the standard iris dataset available
iris = datasets.load_iris()

# Here, we will split it into training and test dataset (90-10 ratio).
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)

# Model building is initializing a Model with the correct set of parameters
# and fitting our training dataset.
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Model testing is predicting the values for test dataset
y_predicted = model.predict(X_test)
print(classification_report(y_test, y_predicted))

# Based on the model's metrics, you can either deploy your model or re-train it.

