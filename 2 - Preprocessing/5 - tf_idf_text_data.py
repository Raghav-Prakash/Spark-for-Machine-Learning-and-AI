# Calculating the Term Frequency (TF) and Inverted Document Frequency (IDF) of words from the previous result.

'''
TF -> compute word frequencies in that document (sentence)
IDF -> compute relative word frequencies among all documents (corpus of 3 sentences)
'''

# import the necessary libraries (for TF and IDF)
from pyspark.ml.feature import HashingTF, IDF

# displaying the previous result's dataframe which we'll be using as input to compute the TF-IDF.
sent_tokenized_df.show()
'''
+---+--------------------+--------------------+
| id|            sentence|               words|
+---+--------------------+--------------------+
|  1|This is an introd...|[this, is, an, in...|
|  2|MLlib includes li...|[mllib, includes,...|
|  3|It also contains ...|[it, also, contai...|
+---+--------------------+--------------------+
'''

# 1. create a HashingTF object with inputCol and outputCol arguments.
# 2. since implicit fitting takes place, transform our input dataframe 

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
sent_hfTF_df = hashingTF.transform(sent_tokenized_df) # hfTF -> hashing function Term Frequency

# displaying the first row of the resultant data frame.
sent_hfTF_df.take(1)
'''
[Row(id=1, 
sentence='This is an introduction to Spark MLlib', 
words=['this', 'is', 'an', 'introduction', 'to', 'spark', 'mllib'], 
rawFeatures=SparseVector(20, {1: 2.0, 5: 1.0, 6: 1.0, 8: 1.0, 12: 1.0, 13: 1.0}))]

The word "this" is mapped to index 1 in rawFeatures SparseVector, "is" mapped to index 5 and so on.
'''

# After computing the TF, we compute the IDF using the above dataframe as input.

# 1. create an IDF object with inputCol and outputCol arguments
# 2. fit the input dataframe to our model object created above
# 3. transform the input data frame into the resultant data frame using the object created above

idf = IDF(inputCol="rawFeatures", outputCol="idf_features")
idf_model = idf.fit(sent_hfTF_df)
tf_idf_df = idf_model.transform(sent_hfTF_df)

# showing the first row
tf_idf_df.take(1)
'''
[Row(id=1, 
sentence='This is an introduction to Spark MLlib', 
words=['this', 'is', 'an', 'introduction', 'to', 'spark', 'mllib'], 
rawFeatures=SparseVector(20, {1: 2.0, 5: 1.0, 6: 1.0, 8: 1.0, 12: 1.0, 13: 1.0}), 
idf_features=SparseVector(20, {1: 0.5754, 5: 0.6931, 6: 0.2877, 8: 0.2877, 12: 0.0, 13: 0.2877}))]

The values in the SparseVector of the idf_features show the relative frequency of words in our corpus
of 3 sentences.
'''
