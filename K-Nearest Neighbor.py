#import library : 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from matplotlib.colors import ListedColormap
# import the dataset : 
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# Split The Dataset To Train And Test Sets : 
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2,random_state=1)
# Implementing The Feature Scalling :  
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# Create The Model : 
classifier = KNeighborsClassifier(n_neighbors= 5 , metric='euclidean' , p=2 )
classifier.fit(x_train , y_train)
# Predicting New Result : 
print(classifier.predict(sc.transform([[30,87000]])))
# Predicting The Test Result : 
y_pred = classifier.predict(x_test)
# Making Confusion Matrix : 
cm = confusion_matrix(y_test , y_pred)
accuracy_score(y_test , y_pred)

if __name__ == '__main__' : 
    num = int(input("Enter What do you want to visluaize ?"))
    
    if (num == 1) : 
        X_set, y_set = sc.inverse_transform(x_train), y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                            np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
        plt.title('K-NN (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show() 
    elif (num == 2) : 
        X_set, y_set = sc.inverse_transform(x_train), y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                            np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
        plt.title('K-NN (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()