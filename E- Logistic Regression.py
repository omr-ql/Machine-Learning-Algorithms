#import library : 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix , accuracy_score
from matplotlib.colors import ListedColormap
# Import Dataset : 
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# Split The Data Training Set And Train set 
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state= 1)
# Implement The Feature : 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
# Create The Model : 
classifier = LogisticRegression(random_state= 0)
classifier.fit(x_train , y_train)
# Predicting A New Result : 
print(classifier.predict(sc.transform([[30,87000]])))
# Predicting The Test Set Result :
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Making The Confusion Matrix :  
cm = confusion_matrix(y_test , y_pred)
accuracy_score(y_test , y_pred)

if __name__ == '__main__' : 
    num = int(input("Do you want to print the test model or train model : "))
    if (num == 1) : 
    # Visualize the Test Model :    
        x_set, y_set = sc.inverse_transform(x_train), y_train
        x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                            np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
        plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
                    alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
        plt.title('Logistic Regression (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
    elif (num == 2) : 
    # Visualize the Train Model :
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                            np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
        plt.title('Logistic Regression (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()