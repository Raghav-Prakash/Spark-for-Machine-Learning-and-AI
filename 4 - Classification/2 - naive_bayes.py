# Naive-Bayes Classification and its evaluation

# import the necessary libraries
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# continuing with the pre-processed dataframe, this will be the input
iviris_df.show(1)
'''
+------------+-----------+------------+-----------+-----------+-----------------+-----+
|sepal_length|sepal_width|petal_length|petal_width|    species|         features|label|
+------------+-----------+------------+-----------+-----------+-----------------+-----+
|         5.1|        3.5|         1.4|        0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|  0.0|
+------------+-----------+------------+-----------+-----------+-----------------+-----+
'''

# display the number of rows in the input data frame
iviris_df.count()
# 150

'''
Given this dataframe, we split the 150 rows into two dataframes - training and testing.
We take (approximately) 60% of the data into training and the remaining 40% into testing.
We start the splitting process with the first input row (seed=1)
'''
splits = iviris_df.randomSplit([0.6, 0.4],1)
train_df = splits[0]
test_df = splits[1]

# displaying the number of rows in the training and testing data frames.
train_df.count()
# 92
test_df.count()
# 58

'''
using an instance of NaiveBayes with the model type as "multinomial" (since there are 3 nominal labels
(species) to predict), we fit the training data frame into our model.
'''
nb = NaiveBayes(modelType="multinomial")
nb_model = nb.fit(train_df)

# using the model, we transform the testing data to get the predicitions, which will be the resultant data frame
predictions_df = nb_model.transform(test_df)

# display the contents of the predicitions data frame
predictions_df.show(1)
'''
+------------+-----------+------------+-----------+-----------+-----------------+-----+--------------------+--------------------+----------+
|sepal_length|sepal_width|petal_length|petal_width|    species|         features|label|       rawPrediction|         probability|prediction|
+------------+-----------+------------+-----------+-----------+-----------------+-----+--------------------+--------------------+----------+
|         4.5|        2.3|         1.3|        0.3|Iris-setosa|[4.5,2.3,1.3,0.3]|  0.0|[-10.360506349494...|[0.56204387804619...|       0.0|
+------------+-----------+------------+-----------+-----------+-----------------+-----+--------------------+--------------------+----------+

So the new columns are rawPredicition, probability and predicition. 
'''

'''
The predicition is 0.0 and the label for that row is also 0.0, meaning that it correctly predicted the species
for that row. But it's not necessary for it to correctly predict the species of all rows. 
So we evaluate this model using the "label" column, "predicition" column and the metric type as "accuracy".
'''

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nb_accuracy = evaluator.evaluate(predictions_df)

# display the accuracy of our predicition
nb_accuracy 
# 0.5862068965517241

# So the predicition accuracy is only around 59%. 
