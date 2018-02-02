# Clustering using K-Means

# import the necessary libraries
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler # needed to store data into a list for later clustering
from pyspark.ml.clustering import KMeans

# read in the csv file
cluster_df = spark.read.csv("/Users/raghav/Documents/Projects/Spark\ for\ Machine\ Learning\ and\ AI/3-clustering/clustering_dataset.csv", header=True, inferSchema=True)
# inferSchema is required since we're working with numerical data

# since there are 75 rows in the table, we'll display all 75 rows.
cluster_df.show(75)
'''
+----+----+----+
|col1|col2|col3|
+----+----+----+
|   7|   4|   1|
|   7|   7|   9|
|   7|   9|   6|
|   1|   6|   5|
|   6|   7|   7|
|   7|   9|   4|
|   7|  10|   6|
|   7|   8|   2|
|   8|   3|   8|
|   4|  10|   5|
|   7|   4|   5|
|   7|   8|   4|
|   2|   5|   1|
|   2|   6|   2|
|   2|   3|   8|
|   3|   9|   1|
|   4|   2|   9|
|   1|   7|   1|
|   6|   2|   3|
|   4|   1|   9|
|   4|   8|   5|
|   6|   6|   7|
|   4|   6|   2|
|   8|   1|   1|
|   7|   5|  10|
|  17|  25|  21|
|  15|  23|  32|
|  42|  25|  45|
|  41|  47|  21|
|  37|  20|  27|
|  40|  18|  26|
|  41|  28|  50|
|  32|  25|  40|
|  24|  29|  35|
|  47|  18|  47|
|  36|  42|  45|
|  49|  29|  15|
|  47|  39|  22|
|  38|  27|  25|
|  45|  23|  40|
|  23|  36|  19|
|  47|  40|  50|
|  37|  30|  40|
|  42|  48|  41|
|  29|  31|  21|
|  36|  39|  48|
|  50|  24|  31|
|  42|  44|  37|
|  37|  39|  46|
|  22|  40|  30|
|  17|  29|  41|
|  85| 100|  69|
|  68|  76|  67|
|  76|  70|  93|
|  62|  66|  91|
|  83|  93|  76|
|  95|  72|  63|
|  75|  94|  95|
|  83|  72|  80|
|  93|  87|  76|
|  86|  93|  63|
|  97|  82|  75|
|  61|  74|  74|
|  84|  90| 100|
|  77|  67|  97|
|  61|  82|  73|
|  81|  60|  69|
|  67|  80|  98|
|  94|  82|  60|
|  69|  73|  74|
|  74|  96|  80|
|  86|  62|  61|
|  88|  68|  95|
|  99|  67|  80|
|  76|  95|  70|
+----+----+----+

There are 3 clusters from the given input data frame:
Cluster 1: range(0-10)
Cluster 2: range(15-60)
Cluster 3: range(60-100)
'''

'''
Before going into clustering, we need the VectorAssembler class to create a feature vector for the clustering
algorithm to work on this data.
'''

# 1. create a VectorAssembler object with inputCols and outputCol as arguments
# 2. transform the input dataframe using the object created above to get the resultant data frame with the feature vector.
# This dataframe with the feature vector will be the input data frame to the next step, KMeans.

vector_assembler = VectorAssembler(inputCols=["col1","col2","col3"],outputCol="feature")
vcluster_df = vector_assembler.transform(cluster_df)

vcluster_df.show()
'''
+----+----+----+--------------+
|col1|col2|col3|      features|
+----+----+----+--------------+
|   7|   4|   1| [7.0,4.0,1.0]|
|   7|   7|   9| [7.0,7.0,9.0]|
|   7|   9|   6| [7.0,9.0,6.0]|
|   1|   6|   5| [1.0,6.0,5.0]|
|   6|   7|   7| [6.0,7.0,7.0]|
|   7|   9|   4| [7.0,9.0,4.0]|
|   7|  10|   6|[7.0,10.0,6.0]|
|   7|   8|   2| [7.0,8.0,2.0]|
|   8|   3|   8| [8.0,3.0,8.0]|
|   4|  10|   5|[4.0,10.0,5.0]|
|   7|   4|   5| [7.0,4.0,5.0]|
|   7|   8|   4| [7.0,8.0,4.0]|
|   2|   5|   1| [2.0,5.0,1.0]|
|   2|   6|   2| [2.0,6.0,2.0]|
|   2|   3|   8| [2.0,3.0,8.0]|
|   3|   9|   1| [3.0,9.0,1.0]|
|   4|   2|   9| [4.0,2.0,9.0]|
|   1|   7|   1| [1.0,7.0,1.0]|
|   6|   2|   3| [6.0,2.0,3.0]|
|   4|   1|   9| [4.0,1.0,9.0]|
+----+----+----+--------------+

Using this, we proceed to KMeans clustering.
'''

# 1. create a KMeans object with K set to 3 (3 clusters)
# 2. Set a seed to start with a particular cluster for faster clustering
# 3. fit the input data frame (output from above) into the kmeans model using the object created above.

k_means = KMeans().setK(3)
k_means = k_means.setSeed(1)
k_model = k_means.fit(vcluster_df)

# get the center feature-vectors for each cluster
centers = k_model.clusterCenters()

# display the centers
centers
'''
[array([ 35.88461538,  31.46153846,  34.42307692]), 
array([ 5.12,  5.84,  4.84]), 
array([ 80.        ,  79.20833333,  78.29166667])]
'''

'''
the first cluster is near the points-vector 35,31,34 (range(15-60))
the second cluster is near the points-vector 5,5,4 (range(0-10))
the third cluster is near the points-vector 80,79,78 (range(60-100))
'''
