# Python Machine Learning tutorial: Support Vector Machines
# Kaique Guimarães Cerqueira

import sklearn # All Machine learning algorithm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# Importing dataset
cancer_data = datasets.load_breast_cancer()

print(cancer_data.feature_names)
print(cancer_data.target_names)

# Separating labels and features, test and train data
X = cancer_data.data
Y = cancer_data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

algorithm = svm.SVC()
algorithm.fit(x_train, y_train)
y_pred = algorithm.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

# Print data -> how to do it with multiple parameters, often not so much human relatable between each other?