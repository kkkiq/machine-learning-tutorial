# Python Machine Learning tutorial: K Nearest Neighbor
# Kaique Guimarães Cerqueira

import pandas as pd # Handle datasets
import numpy as np # Numerical computing
#import pickle   # Save model
import sklearn  # All Machine learning algorithm
from sklearn import linear_model, preprocessing 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

# Load dataframe
data = pd.read_csv("KNN\car.data")
#print(data.head())

# Preprocess data and transform non-numerical attributes into num
labelEncoder = preprocessing.LabelEncoder()
buying = labelEncoder.fit_transform(list(data['buying']))
maintenance = labelEncoder.fit_transform(list(data['maint']))
doors = labelEncoder.fit_transform(list(data['doors']))
persons = labelEncoder.fit_transform(list(data['persons']))
lug_boot = labelEncoder.fit_transform(list(data['lug_boot']))
safety = labelEncoder.fit_transform(list(data['safety']))
classification = labelEncoder.fit_transform(list(data['class']))

predict = "class" # Variable to be predicted

X = list(zip(buying, maintenance, doors, persons, lug_boot, safety)) # features
Y = list(classification) # label -> what we predict

# Separate train and test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Define and evaluate KNN model
algorithm = KNeighborsClassifier(n_neighbors=7)
algorithm.fit(x_train, y_train)
accuracy = algorithm.score(x_test, y_test)
print('Accuracy: ', accuracy)

# Testing the model
prediction = algorithm.predict(x_test)
class_values = ('unacc', 'acc', 'good', 'vgood')

for i in range(len(prediction)):
    print('Predicted: ', class_values[prediction[i]], '\tFeatures: ', x_test[i], '\tActual: ', class_values[y_test[i]])

# Print data -> TODO