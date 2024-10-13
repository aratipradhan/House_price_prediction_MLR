# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the data set
data=pd.read_csv(r"C:\Users\arati\DATAS SCIENCE NIT\SEPTEMBER\26th- mlr\MLR\House_data.csv")

# devide dependent and independent
# we take for prediction bedrooms,sqft,floors
x=data.iloc[:,[3,5,7,10]].values
y=data.iloc[:,2:3].values


# split the data training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# scalling
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# pass the x_train,y_train to regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#  prediction on test value
y_pred=regressor.predict(x_test)
print(y_pred)

#slope
m=regressor.coef_
print(m)

# intercept
c=regressor.intercept_
print(c)


# Save the trained model to disk
import pickle
filename = 'house price.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as house price.pkl")

import os
print(os.getcwd())
