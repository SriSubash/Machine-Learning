from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName('KMeansExample').getOrCreate()
dataset = spark.read.csv('your_csv_file', header=True, inferSchema=True)
assembler = VectorAssembler(inputCols=['label1', 'label2','label3'], outputCol='features')
dataset = assembler.transform(dataset)
kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(dataset)
predictions = model.transform(dataset)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance = {silhouette}")

centers = model.clusterCenters()
print("Cluster Centers: ")

for center in centers:
    print(center)
  
spark.stop()
