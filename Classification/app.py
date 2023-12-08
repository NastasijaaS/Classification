from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, LogisticRegression, NaiveBayes, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum,stddev_samp,expr,udf
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.dataframe import DataFrame

from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics

import matplotlib.pyplot as plt

################ Samo zbog jednostavnog pokretanja #####################

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

########################################################################

PREPROCESS_DATA = True
columns = ["DistanceFromHome", "DistanceFromLastTransaction", "RatioToMedianPurchasePrice", "RepeatRetailer", "UsedChip", "UsedPinNumber", "OnlineOrder"]
columns_scaled = ["DistanceFromHome_S", "DistanceFromLastTransaction_S", "RatioToMedianPurchasePrice_S", "RepeatRetailer_S", "UsedChip_S", "UsedPinNumber_S", "OnlineOrder_S"]


def classificator_evaluation(PredictionAndLabels, output_file):
    
    Metrics = MulticlassMetrics(PredictionAndLabels)
    ConfusionMatrix = Metrics.confusionMatrix()
    cm = ConfusionMatrix.toArray()
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    P = TP + FP
    N = FN + TN
    Ukupno = P + N

    accuracy = (TP + TN) / cm.sum()
    error_rate = 1 - accuracy
    sensitivity = TP / P
    specificity = TN / N 
    precision = (TP) / (TP + FP)
    recall = TP / P
    F = (2 * precision * recall) / (precision + recall)
    pe = (TP + FP) * (TP + FN) / N * N + (TN + FN) * (TN + FP) / N * N  # Kappa statistika

    with open(output_file, 'w') as file:
        sys.stdout = file
        print("         Rezultat klasifikacije        ")
        print("             Da      Ne          Ukupno")
        print(f"Da         {TP}    {FN}          {P}")
        print(f"Ne         {FP}    {TN}          {N}")
        print(f"Ukupno     {P}    {N}         {Ukupno}")
        print("\n\n\n")
        print("Number of instances: ", Ukupno)
        print("Accuracy: \t\t", accuracy)
        print("Error rate: \t", error_rate)
        print("Sensitivity: \t", sensitivity)
        print("Specificity: \t", specificity)
        print("Precision: \t\t", precision)
        print("Recall: \t\t", recall)
        print("F-score: \t\t", F)
        print("Kappa: \t\t\t", pe)
        print("Confusion Matrix:")
        for row in cm:
            print(" ".join(str(int(x)) for x in row))

    sys.stdout = sys.__stdout__


def preprocess_data(data: DataFrame):
    data.show()

    # Izbacujemo duplikate redova:
    duplicate_count = data.dropDuplicates().count() - data.count()
    print(f"\nNumber of duplicate rows: {duplicate_count}\n")

    print("\nKoliko polja u svakoj koloni su null?")
    data.select([sum(col(column).isNull().cast("int")).alias(column) for column in data.columns]).show()

    print("\nDa li vrednosti atributa class balansirane?")
    data.groupBy("fraud").count().show()

    # Balansiranje podataka: (undersampling)
    major_df = data.filter(col("fraud") == 0.0)
    minor_df = data.filter(col("fraud") == 1.0)
    ratio = int(major_df.count()/minor_df.count())
    data = major_df.sample(False, 1/ratio)
    data = data.unionAll(minor_df)

    # Filtriramo opet kolone iz smanjenog dataset-a po klasama:
    major_df = data.filter(col("fraud") == 0.0)
    minor_df = data.filter(col("fraud") == 1.0)

    # Koliko puta smanjujemo dataset:
    # Nakon prethodne transformacije podataka, imali smo oko 85.000 podataka za obe klase, pa se javljao problem
    # da podaci premasuju velicinu heap-a, tako da ovde definisemo koliko puta zelimo da smanjimo dataset;
    # Naravno, jednako smanjujemo i klasu 0.0 i klasu 1.0, kako bi podaci ostali izbalansirani;
    ratio = 16

    data = major_df.sample(False, 1/ratio)
    data2 = minor_df.sample(False, 1/ratio)
    data = data.unionAll(data2)
    
    # Posto su nule do polovine, a 1 od polovine, mesamo redove po random principu:
    data = data.select("*").orderBy(F.rand())
    
    print("\nBalansirani podaci:")
    data.groupBy("fraud").count().show()

    # Izvlacimo mean i standardnu devijaciju za sve podatke;
    # Izbacujemo "outliers", podatke koji previse odskacu od proseka;
    statistics = data.select([stddev_samp(column).alias(column) for column in columns]).first()

    # Definisemo donju i gornju granicu podataka;
    lower_bounds = [(statistics[column] - 2 * statistics[column]) for column in columns]
    upper_bounds = [(statistics[column] + 2 * statistics[column]) for column in columns]

    # Izbacujemo redove koji imaju vrednosti koje previse odskacu;
    data.filter(expr(" AND ".join([f"({column} >= {lower_bound} AND {column} <= {upper_bound})" for column, lower_bound, upper_bound in zip(columns, lower_bounds, upper_bounds)]))).show()

    unlist = udf(lambda x: round(float(list(x)[0]),15), DoubleType())
    for i in columns:
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")                                  # Pretvaramo kolone u vektore;
        scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_S")                                     # Objekat koji vrsi skaliranje;
        pipeline = Pipeline(stages=[assembler, scaler])                                                 # Kreiramo pipeline za ulazne podatke i objekta za skaliranje
        data = pipeline.fit(data).transform(data).withColumn(i+"_S", unlist(i+"_S")).drop(i+"_Vect")    # Dodajemo nove kolone na tabelu, sa zavrsetkom: "_S" (skalirano);

    # Izbacujemo prve 7 kolone, ne mozemo da im zamenimo vrednosti jer je struktura tipe READ_ONLY, pa smo dodali nove kolone sa normalizovanim podacima
    drop_columns = data.columns[:7]
    data = data.drop(*drop_columns)
    data = data.select(*([col(c) for c in data.columns[1:]] + [col(data.columns[0])])) # Kolona Fraud koja oznacava klasu entiteta se pomera kao zadnja kolona

    print("\nNormalizovani podaci:")
    data.show()

    return data   


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .config("spark.executor.memoryOverhead", '2g') \
        .config("spark.task.cpus", "4") \
        .config("autoBroadcast") \
        .appName("Fraud Classificator") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    path = "./card_transdata.csv"

    # 1) Ucitavanje podataka:
    raw_data: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)

    # 2) Preprocesiranje:
    if (PREPROCESS_DATA):
        data = preprocess_data(raw_data)

    # 3) Kreiranje modela:
    # Pretvaramo kolonu 'Fraud' u 'label';
    indexer = StringIndexer(inputCol=data.columns[-1], outputCol="label")
    indexedData = indexer.fit(data).transform(data)

    # Odabir relevantnih atributa i pretvaranje u vektor;
    assembler = VectorAssembler(inputCols=columns_scaled if PREPROCESS_DATA else columns, outputCol="features")
    assembledData = assembler.transform(indexedData)

    # Deljenje podataka na skup za treniranje i skup za testiranje;
    (trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=58137129) 

    lr = LogisticRegression()
    nb = NaiveBayes()
    svm = LinearSVC(labelCol="label", maxIter=10)
    dt = DecisionTreeClassifier()
    gbt = GBTClassifier()
    rf = RandomForestClassifier()


    lrModel = lr.fit(trainingData) 
    # print("\nLR zavrsen!")
    nbModel = nb.fit(trainingData) 
    # print("\nNB zavrsen!")
    svmModel = svm.fit(trainingData)
    # print("\nLinearSVC zavrsen!")
    dtModel = dt.fit(trainingData)
    # print("\nDT zavrsen!")
    gbtModel = gbt.fit(trainingData)
    # print("\nGBT zavrsen!")
    rfModel = rf.fit(trainingData)
    # print("\nRF zavrsen!")

    print("\nTreniranje svih modela zavrseno!\n")


    # 4) Testiranje kreiranih modela:

    lrPredictions = lrModel.transform(testData)
    nbPredictions = nbModel.transform(testData)
    svmPredictions = svmModel.transform(testData)
    dtPredictions = dtModel.transform(testData)
    gbtPredictions = gbtModel.transform(testData)
    rfPredictions = rfModel.transform(testData)

    predictions = [lrPredictions, nbPredictions, svmPredictions, dtPredictions, gbtPredictions, rfPredictions]

    print("\nTestiranje svih modela zavrseno!")

    
    # Evaluacija modela
    lrEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    nbEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    svmEvaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    dtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    gbtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    rfEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    lrAccuracy = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "accuracy"})
    nbAccuracy = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "accuracy"})
    svmAccuracy = svmEvaluator.evaluate(svmPredictions)
    dtAccuracy = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "accuracy"})
    gbtAccuracy = gbtEvaluator.evaluate(gbtPredictions, {gbtEvaluator.metricName: "accuracy"})
    rfAccuracy = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: "accuracy"})

    lrPredictionAndLabels = lrPredictions.select("prediction", "label").rdd
    classificator_evaluation(lrPredictionAndLabels,"Logistic Regression.txt")
    nbPredictionAndLabels = nbPredictions.select("prediction", "label").rdd
    classificator_evaluation(nbPredictionAndLabels,"Naive Bayes.txt")
    svmPredictionAndLabels = svmPredictions.select("prediction", "label").rdd
    classificator_evaluation(svmPredictionAndLabels,"SVM.txt")
    dtPredictionAndLabels = dtPredictions.select("prediction", "label").rdd
    classificator_evaluation(dtPredictionAndLabels,"Decision Tree.txt")
    gbtPredictionAndLabels = gbtPredictions.select("prediction", "label").rdd
    classificator_evaluation(gbtPredictionAndLabels,"GBT.txt")                                          
    rfPredictionAndLabels = rfPredictions.select("prediction", "label").rdd
    classificator_evaluation(rfPredictionAndLabels, "RF.txt")


    print("\nEvaluacija svih modela zavrsena!")

    # 5) Cuvanje rezultata u txt fajlovima:
    
    print("\n\nLogistic Regression Accuracy: " + str(lrAccuracy))
    print("Naive Bayes Accuracy: " + str(nbAccuracy))
    print("SVM Accuracy: " + str(svmAccuracy))
    print("Decision Tree Accuracy: " + str(dtAccuracy))
    print("Gradient Boosted Trees: " + str(gbtAccuracy))
    print("Random Forest: " + str(rfAccuracy))

    # 6) Prikaz grafa preciznosti:

    fig, ax = plt.subplots()

    algorithms = ['LR', 'NB', 'SVM', 'DT', 'GBT', 'RF']
    precissions = [lrAccuracy, nbAccuracy, svmAccuracy, dtAccuracy, gbtAccuracy, rfAccuracy]
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:red']

    ax.bar(algorithms, precissions, color=bar_colors)

    ax.set_ylabel('precission')
    ax.set_title('Precizost klasifikatora nakon testiranja;')

    plt.savefig('final_results.png')
    plt.show()

    spark.stop()