# Linear Regression

# import the necessary libraries
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# read in the csv file
pp_df = spark.read.csv("/Users/raghav/Documents/Projects/Spark\ for\ Machine\ Learning\ and\ AI/5-regression/power_plant.csv", header=True, inferSchema=True)

'''
the csv file above is of a power plant with the columns:
AT - Ambient Temperature
V - exhaust Vacuum
AP - Ambient Pressure
RH - Relative Humidity
PE - Power Emitted

The csv file contains these column names as headers so we include those (headers=True) and the column values
will be read as string unless we infer the schema (inferSchema=True) so that the values will be read as double. 

The columns AT, V, AP and RH will be used to predict the value of PE using linear regression.
'''

# show the column names along with their datatypes
pp_df 
# DataFrame[AT: double, V: double, AP: double, RH: double, PE: double]

# create a feature vector out of this data frame for regression analysis.
vector_assembler = VectorAssembler(inputCols=["AT","V","AP","RH"], outputCol="PE")
vpp_df = vector_assembler.transform(pp_df)

# display one row of vpp_df data frame
vpp_df.take(1)
# [Row(AT=14.96, V=41.76, AP=1024.07, RH=73.17, PE=463.26, features=DenseVector([14.96, 41.76, 1024.07, 73.17]))]

# 1. create a LinearRegression object with features and a label.
# 2. Fit the above dataframe into the model.
lr = LinearRegression(featuresCol="features", labelCol="PE")
lr_model = lr.fit(vpp_df)

'''
We've performed a linear regression on our dataframe containing the feature vectors. So we would have:
1. Coefficients for the four input parameters (AT,V,AP,RH)
2. y-intercept
3. the root-mean-squared error
'''
lr_model.coefficients
# DenseVector([-1.9775, -0.2339, 0.0621, -0.1581])

lr_model.intercept
# 454.6092744523414

lr_model.summary.rootMeanSquaredError
# 4.557126016749488

'''
the regression line crosses the y-axis (y-intercept) at 455 (approx) and the error is 4.55.
So our linear regression model has a 1% error rate.
'''

# saving our model for future regression analyses. saving it as "lr1.model"
lr_model.save("lr1.model") 
