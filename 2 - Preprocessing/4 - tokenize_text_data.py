# Tokenize strings, i.e. convert a sentence into words (tokens)

# import the necessary library
from pyspark.ml.feature import Tokenizer

# create the input data frame
sentences_df = spark.createDataFrame([(1,"This is an introduction to Spark MLlib"),(2, "MLlib includes libraries for classification and regression"),(3,("It also contains supporting tools for pipelines"))], ["id","sentence"])

# 1. create a Tokenizer object with inputCol and outputCol as arguments.
# 2. transform the input data frame using the object created above. 
# (Just like Bucketizer, Tokenizer does implicit fitting.)

sent_token = Tokenizer(inputCol="sentence", outputCol="words")
sent_tokenized_df = sent_token.transform(sentences_df)

# Display the resultant data frame
sent_tokenized_df.show()
'''
+---+--------------------+--------------------+
| id|            sentence|               words|
+---+--------------------+--------------------+
|  1|This is an introd...|[this, is, an, in...|
|  2|MLlib includes li...|[mllib, includes,...|
|  3|It also contains ...|[it, also, contai...|
+---+--------------------+--------------------+

Each sentence is split into words (tokens).
'''
