# Python Machine Learning tutorial: Linear Regression
# Kaique Guimarães Cerqueira

import pandas as pd # Handle datasets
import numpy as np # Numerical computing
import pickle   # Save model
import sklearn  # All Machine learning algorithm
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt # Plot data MATLAB-like
from matplotlib import style

# Loading data
data = pd.read_csv("linear regression\student\student-por.csv", sep=";")
# Trim data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3" # Variable to be predicted

# Rearrange data
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

'''
# Define model/algorithm to be used
algorithm = linear_model.LinearRegression()

accuracy = 0
# Search a good model -> Done!
while accuracy < 0.95:
    # Separate test data and train data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    # Train model -> Where the magic happens
    algorithm.fit(x_train, y_train)
    # Test model (check accuracy)
    accuracy = algorithm.score(x_test, y_test)
    print("Accuracy: " + str(accuracy))

# Save algorithm
with open('linear regression\portuguese_algorithm.pickle', 'wb') as obj:
    pickle.dump(algorithm, obj)
'''
# Load algorithm
pickle_obj = open('linear regression\portuguese_algorithm.pickle', 'rb')
algorithm = pickle.load(pickle_obj)

# Slope value for each axis of data
print('Coefficients (wn): \n', algorithm.coef_)
# Bias coefficient
print('Intercept (w0): \n', algorithm.intercept_) 

# --------------------------------WARNING-----------------------------------
# This is required for plotting data and following the tutorial. But it is fundamentally WRONG.
# We are using test data that was separated differently than when we obtained the model.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Prediction for each student in test group
g3_predicted = algorithm.predict(x_test)

print("Predicted Grade \t| Data used to predict \t| Real Grade")
for i in range(len(g3_predicted)):
    # Predicted Grade | Data used to give prediction | Real Grade
    print(str(g3_predicted[i]) + "\t|" + str(x_test[i]) + "\t|" + str(y_test[i]))

# Plot results
variables = ("G1", "G2", "studytime", "failures", "absences")
for i in range(len(variables)):
    plt.figure()
    plt.scatter(data[variables[i]], data["G3"])
    plt.xlabel(variables[i])
    plt.ylabel('Final Grade')
    
plt.show()
# END OF FILE