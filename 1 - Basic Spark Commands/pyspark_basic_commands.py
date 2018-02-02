# PySpark commands 

# 1. Read a CSV file
emp_df = spark.read.csv('file_directory/file.csv',header=True)

# 2. Show the structure of the table in the input file
emp_df.printSchema()

# 3. Display all columns in the table
emp_df.columns

# Print the first 5 rows in the table
emp_df.take(5)

# Count the number of rows in the table
emp_df.count()

'''
Take a sample from the given input (containing 1000 rows (from the output of the previous command)).
Parameters: To replace the given input file with the sample file(boolean), 
how much to sample(0.1 -> 10% approximately)
'''
sample_df = emp_df.sample(False,0.1)

# Choosing only employees having salary of 100,000 and above from the emp_df table.(These employees are managers)
emp_mgrs_df = emp_df.filter("salary >= 100000")

# Display the first 20 salaries of the managers
emp_mgrs_df.select("salary").show()
