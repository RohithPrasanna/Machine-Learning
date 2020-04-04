import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Reading the Data
USAhousing = pd.read_csv('USA_Housing.csv')

#Exploratory Data analysis
USAhousing.head()
USAhousing.info()
USAhousing.describe()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.pairplot(USAhousing)
sns.heatmap(USAhousing.corr())
sns.distplot(USAhousing['Price'])


#Splitting the data into feature and label columns
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

#Splitting the data into a training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Training the model
lm = LinearRegression()
lm.fit(X_train,y_train)

#Checking the coefficients
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

#Predicting Test Data
predictions = lm.predict(X_test)

#Evaluating the model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
