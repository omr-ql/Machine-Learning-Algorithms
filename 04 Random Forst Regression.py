# Written By : Omar Abdullah Saeed 
# import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
# import dataset : 
dataset = pd.read_csv('Position Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
# import model 
regressor = RandomForestRegressor(n_estimators= 10 , random_state=0)
regressor.fit(x,y)
# Predicting A New Result
regressor.predict([[6.5]])
# Visualising The Random Forest Regressor  : 
if __name__ == "__main__" :  
    X_grid = np.arange(min(x), max(x), 0.01).reshape(-1,1)
    plt.scatter(x, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
