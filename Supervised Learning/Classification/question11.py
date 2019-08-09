# What is a confusion matrix? Explain it for a 2-class problem.

from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import classification_report

# Let's use our iris dataset.
iris = datasets.load_iris()

# For simplicity, let's just use 2 classes from our data. We will remove all the rows from input data
# where target class is 2
target = pd.DataFrame(iris.target)
rows_to_del = target.loc[target[0] == 2]
target = target.drop(rows_to_del.index)

data = pd.DataFrame(iris.data)
data = data.drop(rows_to_del.index)

# Now, we will split the data into train and test dataset.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.10)

# Let's build a linear SVM classifier.
classifier = svm.SVC(kernel='linear', C=0.05)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_mtx = confusion_matrix(y_test, y_pred)

# Here, we will plot the Confusion Matrix
title = "Confusion Matrix"
fig, ax = plt.subplots()
ax.matshow(confusion_mtx, cmap=plt.cm.Blues)

threshold = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    ax.text(i, j, format(confusion_mtx[i, j], 'd'),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if confusion_mtx[i, j] > threshold else "black")

plt.tight_layout()
plt.show()

# You can get all the metrics from classification_report of the model
print("Classification Report")
print(classification_report(y_test, y_pred))
