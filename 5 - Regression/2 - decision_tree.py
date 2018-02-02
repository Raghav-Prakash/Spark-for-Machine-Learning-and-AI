# Decision-Tree regression and evaluating the error

# import the necessary libraries
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# continuing from the linear_regression program, we have the vpp_df data frame.
vpp_df.take(1)
# [Row(AT=14.96, V=41.76, AP=1024.07, RH=73.17, PE=463.26, features=DenseVector([14.96, 41.76, 1024.07, 73.17]))]

# create a training data and testing data from the above data frame.
splits = vpp_df.randomSplit([0.7,0.3]) # 70% -> training data, 30% -> testing data

train_df = splits[0]
test_df = splits[1]

# 1. create a DecisionTreeRegressor object taking in features and the label.
# 2. model using the training data
# 3. predict using the testing data

dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

'''
since we're dealing with a numeric label (PE) rather than a nominal one (like in classification), 
we evaluate the error rate of the predictions

So the metric here is "rmse" -> root mean-squared error
'''
dt_evaulator = RegressionEvaluator(labelCol="PE", predictionCol="predictions", metricName="rmse")
dt_error = dt_evaulator.evaluate(dt_predictions)

# display the root mean-squared error
dt_error
# 4.528377010708296

#Just like linear regression, decision-tree regression gives a 1% error rate for the predicted PE values.
