from pyspark.sql import SparkSession
from pyspark.ml.feature import 
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

#Starting a spark session
spark = SparkSession.builder.appName('cruise').getOrCreate()

#Reading the Data
df = spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)

#Exploratory Data analysis
df.printSchema()
df.show()
df.describe().show()

#Converting categorical features
df.groupBy('Cruise_line').count().show()
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)
indexed.head(5)

#Using vector assembler to merge multiple columns into a vector column
assembler = VectorAssembler(
  inputCols=['Age',
             'Tonnage',
             'passengers',
             'length',
             'cabins',
             'passenger_density',
             'cruise_cat'],
    outputCol="features")
output = assembler.transform(indexed)
output.select("features", "crew").show()
final_data = output.select("features", "crew")

#train test split of the data
train_data,test_data = final_data.randomSplit([0.7,0.3])

#Creating a linear regression model object and fitting the data
lr = LinearRegression(labelCol='crew')
lrModel = lr.fit(train_data)

#Evaluating the model
test_results = lrModel.evaluate(test_data)
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))
