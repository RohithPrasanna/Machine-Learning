import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix


#Reading the Data
yelp = pd.read_csv('yelp.csv')

#Exploratory Data analysis
yelp.head()
yelp.describe()
yelp.info()

#Create a new column called "text length" which is the number of words in the text column
yelp['text length'] = yelp['text'].apply(len)

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
sns.boxplot('stars','text length',data=yelp)
sns.countplot(x='stars',data=yelp)

#Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

#Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)
X = yelp_class['text']
y = yelp_class['stars']

#create a CountVectorizer object
cv = CountVectorizer()

#Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X
X = cv.fit_transform(X)

#Train test split and training the model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
nb = MultinomialNB()
nb.fit(X_train,y_train)

#Predictions and Evaluation of the model
predictions = nb.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

