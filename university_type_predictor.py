import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



#Reading the Data
data = pd.read_csv('College_Data',index_col=0)

#Exploratory Data analysis
data.head()
data.describe()
data.info()

#Exploratory Data Visualisation; Exploring the relationships across the entire data set
sns.scatterplot(x='Grad.Rate',y='Room.Board',data=data,hue='Private')
sns.scatterplot(x='F.Undergrad',y='Outstate',data=data,hue='Private')

#K means model with 2 clusters
kmeans_model = KMeans(n_clusters=2)

#Fitting the model
kmeans_model.fit(data.drop('Private',axis=1))

#Predicting the outcome
predictions = kmeans_model.predict(data.drop('Private',axis=1))

#Cluster center vectors
print(kmeans_model.cluster_centers_)
