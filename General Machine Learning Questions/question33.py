# What does it mean to fit a model? How do the hyperparameters relate?

# In this example, we will use the same iris dataset and train Linear Regression model.

from sklearn.linear_model import LinearRegression
from sklearn import datasets

[X_train, y_train] = datasets.load_iris(return_X_y=True)
model = LinearRegression()
model.fit(X_train, y_train)

# After fitting, let's see what do we get as the coefficients and the intercept
print("Model Coefficients are " + str(model.coef_))
print("Model Intercept is " + str(model.intercept_))
