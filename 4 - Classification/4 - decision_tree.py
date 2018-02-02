# Decision-Tree Classification and evaluation of its prediction accuracy 

# import the necessary library
from pyspark.ml.classification import DecisionTreeClassifier

# using the same training data and testing data from the previous classification programs

# 1. create a DecisionTreeClassifier object using features from the ivirus_df dataframe to get the label
# 2. create a model of using the training data
# 3. get the predictions dataframe using the testing data

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

# evaluating the accuracy of predictions

dt_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)

# display the prediction accuracy
dt_accuracy
# 0.9310344827586207

'''
Naive-Bayes classification gave a prediction accuracy of 59%, 
Multi-layer Perceptron of 95% and 
Decision-Tree Classification of 93%.
'''
