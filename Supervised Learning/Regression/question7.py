# What can you interpret from the coefficient estimates?

from sklearn import linear_model
import numpy as np

# Let's use our same training dataset of sum of 2 integers.
input_data = np.random.randint(50, size=(20, 2))
input_sum = np.zeros(len(input_data))
for row in range(len(input_data)):
    input_sum[row] = input_data[row][0] + input_data[row][1]

# Let's build a simple Linear Regression model
regression_model = linear_model.LinearRegression()
regression_model.fit(input_data, input_sum)

# As you would expect, the coefficients of the inputs should be 1 and intercept 0.
print("Model Coefficients are " + str(regression_model.coef_))
print("Model Intercept is " + str(regression_model.intercept_))

# Since, we had fit_intercept term as True in our model, we get a very small, close to 0 value for it.
# Let's see what happens when we set it to False
regression_model = linear_model.LinearRegression(fit_intercept=False)
regression_model.fit(input_data, input_sum)
print("Model Coefficients are " + str(regression_model.coef_))
print("Model Intercept is " + str(regression_model.intercept_))
