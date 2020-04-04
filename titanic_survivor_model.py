import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Reading the Data
train = pd.read_csv('titanic_train.csv')

#Exploratory Data analysis
train.head()
train.info()
train.describe()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

#Data Cleaning by filling nan values using impute function
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return 
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

#Converting categorical features
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)


#Splitting the data into feature and label columns
X = train.drop('Survived',axis=1)
y = train['Survived']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


#Predicting Test Data
predictions = logmodel.predict(X_test)

#Evaluating the model
print(classification_report(y_test,predictions))
