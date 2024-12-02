from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import when

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("PySpark SQL Server Connection").config(
    "spark.jars", "/usr/share/java/mysql-connector-java-9.1.0.jar").getOrCreate()

# Define MySQL connection parameters
jdbc_url = "jdbc:mysql://192.168.0.110/auto_mpg"
connection_properties = {"user": "nighthawksdb",
                         "password": "sunlightsam829", "driver": "com.mysql.cj.jdbc.Driver"}

# Load the data from MySQL into a Spark DataFrame
df = spark.read.jdbc(url=jdbc_url, table="auto_mpg", properties=connection_properties)

# Change data type of horsepower from string to int
df=df.withColumn("horsepower",col("horsepower").cast("int"))

from pyspark.ml.feature import Imputer
imputer = Imputer(
    inputCols=["horsepower"],  # Column(s) to fill
    outputCols=["horsepower"],  # Column(s) to store filled values
    strategy="mean"  # Strategy: 'mean' or 'median'
)

df_imputed=imputer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=["cylinders", "displacement", "horsepower",
                            "weight", "acceleration", "model year", "origin"], outputCol='features')

df_transformed=assembler.transform(df_imputed)

df_transformed.select('features','mpg')

final_data=df_transformed.select('features','mpg')

train_data,test_data=final_data.randomSplit([0.7,0.3])

lr=LinearRegression(featuresCol="features",labelCol="mpg")

trained_lr_model=lr.fit(train_data)

results=trained_lr_model.evaluate(train_data)

# Print the evaluation metrics
print("Train data Root Mean Squared Error (RMSE): {}".format(results.rootMeanSquaredError))
print("Train data Mean Absolute Error (MAE): {}".format(results.meanAbsoluteError))
print("Train data R-squared (R²): {}".format(results.r2))

unlabeled_data=test_data.select("features")

predictions=trained_lr_model.transform(unlabeled_data)
predictions.show()

test_results=trained_lr_model.evaluate(test_data)

# Print the evaluation metrics
print("Test data Root Mean Squared Error (RMSE): {}".format(test_results.rootMeanSquaredError))
print("Test data Mean Absolute Error (MAE): {}".format(test_results.meanAbsoluteError))
print("Test data R-squared (R²): {}".format(test_results.r2))