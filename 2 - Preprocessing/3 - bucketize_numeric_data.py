# Bucketize numeric data

# import the library
from pyspark.ml.feature import Bucketizer

# create a list of splitting-boundaries for numeric data to be classified and bucketized.
'''
The split list below says the following ranges for each bucket:
bucket 0: range(-infinity, -10.0)
bucket 1: range(-10.0, 0.0)
bucket 2: range(0.0, 10.0)
bucket 3: range(10.0, infinity)
'''
split = [-float("inf"), -10.0, 0.0, 10.0, float("inf")]

# create a list of rows wherein each row has 1 value and then create the input data frame.
b_data = [(-800.0,),(-10.5,),(-1.7,),(0.0,),(8.2,),(90.1,)]
b_df = spark.createDataFrame(b_data,["features"])

# display the input data frame as is.
b_df.show()
'''
+--------+
|features|
+--------+
|  -800.0|
|   -10.5|
|    -1.7|
|     0.0|
|     8.2|
|    90.1|
+--------+
'''

'''
1. create a Bucketizer object which takes as parameters: the splits, inputCol and outputCol.
There's no need to fit as the splits parameter does the fitting implicitly.
2. so we directly transorm the input data frame using the object created above to get the resultant data frame.
'''

bucketizer = Bucketizer(splits=split, inputCol="features", outputCol="bfeatures")
bucketed_df = bucketizer.transorm(b_df)

bucketed_df.show()
'''
+--------+---------+
|features|bfeatures|
+--------+---------+
|  -800.0|      0.0|
|   -10.5|      0.0|
|    -1.7|      1.0|
|     0.0|      2.0|
|     8.2|      2.0|
|    90.1|      3.0|
+--------+---------+

So the values -800.0 and -10.5 are in the first bucket (bucket 0) since they lie between -infinity and -10.0
Similarly, -1.7 lies in the second bucket (bucket 1), 0.0 and 8.2 in the third bucket (bucket 2) and 90.1
in the fourth bucket (bucket 3).
'''
