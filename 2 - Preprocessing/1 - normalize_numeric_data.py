# Normalizing data. Scaling regular data in the range 0 to 1.

# import the necessary libraries.
from pyspark.ml.feature import MinMaxScaler # MinMaxScaler is responsible for normalizing data.
from pyspark.ml.linalg import Vectors # Vectors holds data and this comes from the linear algebra library

# creating data in a data frame. The first argument lists the row values and the second the column names.
features_df = spark.createDataFrame([(1,Vectors.dense([10.0,10000.0,1.0]),),(2,Vectors.dense([20.0,30000.0,2.0]),),(3,Vectors.dense([30.0,40000.0,3.0]),)],["id","features"])

# 1.create a MinMaxScaler object. Convert "inputCol" with regular values to "outputCol" with normalized values.
# 2.fit the input data into our MinMaxScaler object.
# 3.transform the model we get after fitting to get a new data frame with features and sfeatures.

feature_scaler = MinMaxScaler(inputCol="features",outputCol="sfeatures")
smodel = feature_scaler.fit(features_df)
sfeatures_df = smodel.transform(features_df)

# show the first row to check our result
sfeatures_df.take(1)
#output: [Row(id=1, features=DenseVector([10.0, 10000.0, 1.0]), sfeatures=DenseVector([0.0, 0.0, 0.0]))]

# Display the entire dataframe to compare the original values and their equivalent normalized values.
sfeatures_df.select("features","sfeatures").show() 

'''
final output:
+------------------+--------------------+
|          features|           sfeatures|
+------------------+--------------------+
|[10.0,10000.0,1.0]|       [0.0,0.0,0.0]|
|[20.0,30000.0,2.0]|[0.5,0.6666666666...|
|[30.0,40000.0,3.0]|       [1.0,1.0,1.0]|
+------------------+--------------------+
'''
