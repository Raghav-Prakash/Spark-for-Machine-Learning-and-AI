# Classification using Multi-layer Perceptrons (a kind of a neural network) and evaluating the accuracy

# import the necessary library
from pyspark.ml.classification import MultilayerPerceptronClassifier

'''
continuing from the naive_bayes classification, we use the viris_df dataframe as input.

We also need layers for the neurons. We have 4 layers - input, middle two layers and output.

Since the input has 4 parametes - sepal_length, sepal_width, petal_length and petal_width - we have 4 sources
of input to predict the type of species. So the first layer has 4 neurons.

Since we have 3 possible outputs - iris_setosa, iris_versicolor and iris_virginica - we have 3 neurons
in the last layer.

The middle two layers have 5 neurons each for better predicition accuracy.
'''
layers = [4,5,5,3]

# 1. we create the neuron object
# 2. we build an mlp model using the training data (from the naive_bayes program)
# 3. we make predictions using the testing data (from the naive_bayes program) and the model 

mlp = MultilayerPerceptronClassifier(layers=layers, 1) # seed for random number generator
mlp_model = mlp.fit(train_df)
mlp_predictions = mlp_model.transform(test_df)

# evaluating the accuracy of the predictions.
mlp_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)

# display the accuracy 
mlp_accuracy
# 0.9482758620689655

'''
The Naive-Bayes classification gave a prediction accuracy of 59% 
while the mulit-layer perceptron classification gave a prediction accuracy of 95%. 

So using neural networks, we get better accuracy.
'''
