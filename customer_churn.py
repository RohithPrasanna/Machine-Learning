import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Reading the Data
data = pd.read_csv('customer_churn.csv')

#Exploratory Data analysis
data.head()
data.describe()
data.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.pairplot(data=data,hue='Age')
sns.jointplot(x='Account_Manager',y='Total_Purchase',data=data)


#Splitting the data into feature and label columns
X = data[['Age','Total_Purchase', 'Account_Manager',
       'Num_Sites']]
y = data['Churn']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the model
model = LogisticRegression()
model.fit(X_train,y_train)


#Predicting Test Data
predictions = model.predict(X_test)

#Evaluating the model
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
