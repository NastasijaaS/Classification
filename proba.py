from pyspark import SparkContext, SparkConf 
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

conf = SparkConf().setAppName("test").setMaster("local")
sc = SparkContext(conf=conf)
data = [1, 2, 3, 4, 50, 61, 72, 8, 9, 19, 31, 42, 53, 6, 7, 23]
rdd = sc.parallelize(data)
filteredRDD = rdd.filter(lambda x: x > 10)
transformedRDD = filteredRDD.map(lambda x: x*x)
sum = transformedRDD.reduce(lambda x, y: x+y)
print("Broj elemenata:", sum)