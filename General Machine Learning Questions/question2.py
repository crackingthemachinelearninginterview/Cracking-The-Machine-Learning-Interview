# How would you differentiate a Machine Learning algorithm from other algorithms?

# Let's train a simple Scikit-Learn Linear model that returns the sum of two numbers
from sklearn import linear_model
import numpy as np

# First, create our training (input) dataset. The input_data has the 2 input integers
# and the input_sum is the resulting sum of them.
input_data = np.random.randint(50, size=(20, 2))
input_sum = np.zeros(len(input_data))
for row in range(len(input_data)):
    input_sum[row] = input_data[row][0] + input_data[row][1]

# Now, we will build a simple Linear Regression model which trains on this dataset.
linear_regression_model = linear_model.LinearRegression(fit_intercept=False)
linear_regression_model.fit(input_data, input_sum)

# Once, the model is trained, let's see what it predicts for the new data.
predicted_sum = linear_regression_model.predict([[60, 24]])
print("Predicted sum of 60 and 24 is " + str(predicted_sum))

# To give you an insight into this model, it predicts the output using the following equation:
# output = <coefficient for 1st number> * < 1st number> + <coefficient for 2nd number> * < 2nd number>

# Now, our model should have 1, 1 as the coefficients which means it figured out that for 2 inout integers,
# it has to return their sum
print("Coefficients of both the inputs are " + str(linear_regression_model.coef_))

# This is, of course, very basic stuff, but I hope you get the idea.
