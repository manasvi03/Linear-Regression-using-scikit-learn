# -*- coding: utf-8 -*-
"""
@author: MANASVI
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#importing the data
data = pd.read_csv("Salary_Data.csv")

#spilliting the data into features: YearsExperience and labels: Salary
X = data.iloc[:,0]
Y = data.iloc[:, 1]

#plotting the data to see pattern
plt.scatter(X,Y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience-Salary pattern")
plt.show()


#data preprocessing
data.isna().sum()         #checking for na values
scaler = StandardScaler()         #normalizing the data
X = scaler.fit_transform(X.values.reshape(-1,1))
Y = scaler.fit_transform(Y.values.reshape(-1,1))

#splitting the data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0, test_size = 0.2 )


#training the model using Linear Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)

#now predicting the values
Y_pred = lr.predict(X_test)

#checking the accuracy of the predicting
print("Accuracy: ", lr.score(X_test, Y_test))    #Accuracy:  0.9881695157291261
print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred))   #Mean Squared Error:  0.017650963976937245







