# Written By : Omar Abdullah Saeed 
#Support Vector Regression 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# Import The Dataset : 
dataset = pd.read_csv('Position Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1) # Make it 2D Array to be able to handle it 
# Apply Feature Scalling : 
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
# Evaluate The Model : 
regressor = SVR(kernel='rbf')
regressor.fit(x,y)
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))
if __name__ == "__main__" : 
    # Visualising The SVR : 
    plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'black')
    plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
    plt.title('SVR Model Prediction')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
