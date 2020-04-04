import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Exploratory Data analysis
customers.head()
customers.describe()
customers.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.jointplot(x = 'Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y = 'Length of Membership', data=customers,kind='hex')
sns.pairplot(data=customers)
sns.lmplot(x='Yearly Amount Spent', y = 'Length of Membership', data=customers)


#Reading the Data
customers = pd.read_csv('Ecommerce Customers')

#Splitting the data into feature and label columns
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the model
model = LinearRegression()
model.fit(X_train,y_train)

#Predicting Test Data
predictions = model.predict(X_test)

#Evaluating the model
MAE = metrics.mean_absolute_error(y_test,predictions)
MSE = metrics.mean_squared_error(y_test,predictions)
RMSE = np.sqrt(MSE)
print('Mean Absolute Error = {}'.format(MAE))
print('Mean Squared Error = {}'.format(MSE))
print('Root Mean Squared Error = {}'.format(RMSE))
