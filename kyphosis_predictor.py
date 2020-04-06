import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Reading the Data
df = pd.read_csv('kyphosis.csv')

#Exploratory Data analysis
df.head()
df.describe()
df.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.pairplot(df,hue='Kyphosis',palette='Set1')


#Splitting the data into feature and label columns
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#Training the decision tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


#Predicting Test Data using decision tree model
predictions = dtree.predict(X_test)

#Training the random forest classifier model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


#Predicting Test Data using decision tree model
rfc_pred = rfc.predict(X_test)

#Evaluating boht the model
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
