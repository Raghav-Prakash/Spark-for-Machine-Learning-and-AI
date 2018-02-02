# gradient-boosted tree regression and evaluating the error

# import the necessary libraries
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# continuing from the decisiont_tree regression program, we have the training and testing data frames.

# 1. create a GBTRegressor object taking in features and a label as parameters.
# 2. create a model using the training data
# 3. get predictions using the testing data

gbt = GBTRegressor(featuresCol="features", labelCol="PE")
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

# evaluating the root-mean-squared error of our predictions.

gbt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")
gbt_error = gbt_evaluator.evaluate(gbt_predictions)

# display the error rate
gbt_error
# 4.080344345768068

# this has a better error rate than the decision-tree regressor and the linear regressor for this dataset.
