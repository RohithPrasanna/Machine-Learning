import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Reading the Data
loans = pd.read_csv('loan_data.csv')

#Exploratory Data analysis
loans.head()
loans.describe()
loans.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.countplot(x='purpose',data=loans,hue='not.fully.paid')
sns.jointplot(x='fico',y='int.rate',data=loans)
sns.lmplot(x='int.rate',y='fico',data=loans,hue='credit.policy',col='not.fully.paid')

#Converting categorical features
cat_feats = ['purpose']
final_data = pd.get_dummies(data=loans,columns=cat_feats,drop_first=True)

#Splitting the data into feature and label columns
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the decision tree model
dtree_model = DecisionTreeClassifier()
dtree_model.fit(X_train,y_train)


#Predicting Test Data using decision tree model
preds_dtree = dtree_model.predict(X_test)

#Training the random forest classifier model
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_train,y_train)


#Predicting Test Data using decision tree model
preds_rfc = rfc_model.predict(X_test)

#Evaluating boht the model
print(confusion_matrix(y_test,preds_dtree))
print(classification_report(y_test,preds_dtree))
print(confusion_matrix(y_test,preds_rfc))
print(classification_report(y_test,preds_rfc))
