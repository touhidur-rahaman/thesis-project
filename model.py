# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# dataset = pd.read_csv('hiring.csv')
dataset = pd.read_csv('bdprop.csv')



dataset['area'].fillna(0, inplace=True)

# dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :5]
# print(X)
# Converting words to integer values
# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]
# print(y)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# linear regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# SVM 
from sklearn.svm import SVC

regressorSVC = SVC()

# neural net
from sklearn.neural_network import MLPClassifier
regressorNN = MLPClassifier()

#Fitting model with trainig data
# regressor.fit(X, y)

# regressor.fit(X_train, y_train)

# svm
# regressorSVC.fit(X_train, y_train)
regressorNN.fit(X_train, y_train)

# predictions = regressor.predict(X_test)

# predictions = regressorSVC.predict(X_test)
predictions = regressorNN.predict(X_test)

# predictions_series = pd.Series(predictions)

# print(pd.Series(predictions))
# accuracy = np.mean(abs(predictions - y_test) / y_test)
# accuracy = abs(predictions - y_test)
error_rate = np.mean(abs(predictions - y_test.to_numpy()) / y_test.to_numpy())

# print(predictions)
# print (y_test.to_numpy())
print (error_rate)

# Saving model to disk
# pickle.dump(regressor, open('apartment-model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('apartment-model.pkl','rb'))
# print(model.predict([[5, 1, 2, 2, 1200]]))

