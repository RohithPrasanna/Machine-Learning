import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

#Reading the Data
iris = sns.load_dataset('iris')

#Exploratory Data analysis
iris.head()
iris.describe()
iris.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.pairplot(iris,hue='species')
setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_length'],setosa['sepal_width'],cmap='plasma')


#Splitting the data into feature and label columns
X = iris.drop('species',axis=1)
y = iris['species']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the decision tree model
svc_model = SVC()
svc_model.fit(X_train,y_train)


#Predicting Test Data using decision tree model
preds_init = svc_model.predict(X_test)


#Evaluating boht the model
print(confusion_matrix(y_test,preds_init))
print(classification_report(y_test,preds_init))

#Using gridsearchCV to fine tune parameters to get an even better outcome
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']} 
svc_grid = GridSearchCV(SVC(),param_grid,verbose=3)
svc_grid.fit(X_train,y_train)
preds_final = svc_grid.predict(X_test)
print(confusion_matrix(y_test,preds_init))
print(classification_report(y_test,preds_init))
