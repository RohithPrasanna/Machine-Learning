from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Starting a spark session
spark = SparkSession.builder.appName('nlp').getOrCreate()

#Reading the Data and cleaning it
data = spark.read.csv("smsspamcollection/SMSSpamCollection",inferSchema=True,sep='\t')
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data = data.withColumn('length',length(data['text']))

#Exploratory Data analysis
data.printSchema()
data.show()
data.describe().show()

#Feature Transformations
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')

#Using vector assembler to merge multiple columns into a vector column
clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

#Creating a pipeline
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)

#train test split of the data
clean_data = clean_data.select(['label','features'])
(training,testing) = clean_data.randomSplit([0.7,0.3])

#Creating a linear regression model object and fitting the data
nb = NaiveBayes()
spam_predictor = nb.fit(training)
test_results = spam_predictor.transform(testing)

#Evaluating the model
test_results = lrModel.evaluate(test_data)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
