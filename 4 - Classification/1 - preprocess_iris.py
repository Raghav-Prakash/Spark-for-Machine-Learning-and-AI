# Preprocessing data for later classification

# import the necessary libraries
from pyspark.sql.functions import * # will be required for renaming columns in the input data frame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer # will be required for indexing the string column in the data frame

# read in the input csv file
iris_df = spark.read.csv("/Users/raghav/Documents/Projects/Spark\ for\ Machine\ Learning\ and\ AI/4-classification/iris.csv", inferSchema=True)

# display the columns and their datatypes
iris_df
# DataFrame[_c0: double, _c1: double, _c2: double, _c3: double, _c4: string]

# and display the first row of values
iris_df.take(1)
# [Row(_c0=5.1, _c1=3.5, _c2=1.4, _c3=0.2, _c4='Iris-setosa')]

# The column names are hard to get. So renaming them using "alias" for better querying.
iris_df = iris_df.select(
	col("_c0").alias("sepal_length"),
	col("_c1").alias("sepal_width"),
	col("_c2").alias("petal_length"),
	col("_c3").alias("petal_width"),
	col("_c4").alias("species"),
)

# and now displaying the columns and their datatypes
iris_df
# DataFrame[sepal_length: double, sepal_width: double, petal_length: double, petal_width: double, species: string]

# and displaying the first row pf values
 iris_df.take(1)
# [Row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa')]

# Now getting the numerical values from the input data frame above into a feature vector using VectorAssembler
vector_assembler = VectorAssembler(inputCols=["sepal_length","sepal_width","petal_length","petal_width"], outputCol="features")
viris_df = vector_assembler.transform(iris_df)

viris_df.take(1)
# [Row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa', 
# features=DenseVector([5.1, 3.5, 1.4, 0.2]))]

# indexing the "species" column using StringIndexer 
'''
label=0 -> the first species "Iris-setosa"
label=1 -> the second species "Iris-versicolor"
label=2 -> the third species "Iris-virginica"
'''
indexer = StringIndexer(inputCol="species", outputCol="label")
iviris_df = indexer.fit(viris_df).transform(viris_df)

# and finally, the first row of the resultant pre-processed data frame:
iviris_df.show(1)
'''
+------------+-----------+------------+-----------+-----------+-----------------+-----+
|sepal_length|sepal_width|petal_length|petal_width|    species|         features|label|
+------------+-----------+------------+-----------+-----------+-----------------+-----+
|         5.1|        3.5|         1.4|        0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|  0.0|
+------------+-----------+------------+-----------+-----------+-----------------+-----+
'''
