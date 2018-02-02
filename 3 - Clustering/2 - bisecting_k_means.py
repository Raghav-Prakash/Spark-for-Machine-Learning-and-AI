# Hierarchical clustering - Bisecting K-Means

# continuing from K-Means clustering program

#import the necessary library 
from pyspark.ml.clustering import BisectingKMeans

# using vcluster_df as the input data frame 
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
'''

# 1. create a BisectingKMeans object and set number of clusters, K, to be 3
# 2. set the seed
# 3. fit the input dataframe into our model using the object created above

bk_means = BisectingKMeans().setK(3)
bk_means = bk_means.setSeed(1)
bk_model = bk_means.fit(vcluster_df)

# get the centers
bk_centers = bk_model.clusterCenters()

# display the center vectors
bk_centers
'''
[array([ 5.12,  5.84,  4.84]), 
array([ 35.88461538,  31.46153846,  34.42307692]), 
array([ 80.        ,  79.20833333,  78.29166667])]
'''

# comparing this with the centers from KMeans
centers
'''
[array([ 35.88461538,  31.46153846,  34.42307692]), 
array([ 5.12,  5.84,  4.84]), 
array([ 80.        ,  79.20833333,  78.29166667])]
'''
