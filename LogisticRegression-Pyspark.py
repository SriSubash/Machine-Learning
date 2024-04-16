from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('LogisticRegressionExample').getOrCreate()
dataset = spark.read.csv('your_csv_file', header=True, inferSchema=True)
assembler = VectorAssembler(inputCols=['label1', 'label2','label3'], outputCol='features')
dataset = assembler.transform(dataset)
dataset = dataset.withColumnRenamed('your_prediction', 'output')
train_data, test_data = dataset.randomSplit([0.7, 0.3], seed=42)

lr = LogisticRegression(featuresCol='features', labelCol='target')
lrModel = lr.fit(train_data)
predictions = lrModel.transform(test_data)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target")
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy}")
print(f"Coefficients: {str(lrModel.coefficients)}")
print(f"Intercept: {str(lrModel.intercept)}")
spark.stop()
