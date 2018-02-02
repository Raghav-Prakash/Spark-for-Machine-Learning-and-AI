# Standardizing data. Converting given input values to value in the range -1 to 1 with mean 0 and variance 1.

# import the necessary libraries.
from pyspark.ml.feature import StandardScaler # StandardScaler is responsible for standardizing input data
from pyspark.ml.linalg import Vectors

# create a data frame with input data.
features_df = spark.createDataFrame([(1,Vectors.dense([10.0,10000.0,1.0]),),(2,Vectors.dense([20.0,30000.0,2.0]),),(3,Vectors.dense([30.0,40000.0,3.0]),)],["id","features"])

'''
1. create a StandardScaler object with the inputCol and outputCol parameters along with two other ones:
	withStd saying the distribution needs a standard deviation and withMean saying it also needs mean, i.e.
	our data is normally distributed. (Bell curve)
2. using the object created above, fit our input data
3. using the object created above, transorm our input data to get the resultant dataframe with standardized values.
'''

features_stand_scaler = StandardScaler(inputCol="features",outputCol="sfeatures",withStd=True,withMean=True)
stand_smodel = features_stand_scaler.fit(features_df)
stand_sfeatures = stand_smodel.transorm(features_df)

stand_sfeatures.select("features","sfeatures").show()
'''
+------------------+--------------------+
|          features|           sfeatures|
+------------------+--------------------+
|[10.0,10000.0,1.0]|[-1.0,-1.09108945...|
|[20.0,30000.0,2.0]|[0.0,0.2182178902...|
|[30.0,40000.0,3.0]|[1.0,0.8728715609...|
+------------------+--------------------+
'''
