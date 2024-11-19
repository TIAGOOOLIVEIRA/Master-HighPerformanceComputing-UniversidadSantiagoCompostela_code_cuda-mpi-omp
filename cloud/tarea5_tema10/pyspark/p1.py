from pyspark.sql import SparkSession
from pyspark.sql.functions import count
import sys

def main(args):
    #spark-submit --master yarn  --num-executors 4 --driver-memory 4g p1.py /user/curso370/data/cite75_99.txt /user/curso370/data/apat63_99.txt /user/curso370/output/cite75_99.parquet /user/curso370/output/apat63_99.parquet
    #hdfs dfs -ls /user/curso370/output/apat63_99.parquet
    
    # Comprueba el número de argumentos
    # sys.argv[1] es el primer argumento, sys.argv[2] el segundo, etc.
    if len(sys.argv) != 5:
        print("Usar: p1.py cite75_99.txt apat63_99.txt dfCitas.parquet dfInfo.parquet")
        exit(-1)

    cite75_99_path = args[1]
    apat63_99_path = args[2]
    output_cite75_99_path = args[3]
    output_apat63_99_path = args[4]

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Practica 1 de Tomas") \
        .getOrCreate()

    # Cambio la verbosidad para reducir el número de
    # mensajes por pantalla
    spark.sparkContext.setLogLevel("FATAL")

    #Load cite75_99.txt
    cite_df = spark.read.option("header", True).csv(cite75_99_path)

    #Group and count
    citation_counts_df = cite_df.groupBy("CITED").agg(count("CITED").alias("ncitas")).withColumnRenamed("CITED","NPatente")

    #Load apat63_99.txt
    apat_df = spark.read.option("header", True).csv(apat63_99_path)
    
    #Select only PATENT, GYEAR, and COUNTRY columns
    selected_apat_df = apat_df.select("PATENT", "GYEAR", "COUNTRY").withColumnRenamed("PATENT","NPatente").withColumnRenamed("GYEAR","Anho").withColumnRenamed("COUNTRY","Pais")

    #Write to Parquet with Gzip compression
    citation_counts_df.write \
        .option("compression", "gzip") \
        .parquet(output_cite75_99_path)

    #Write to Parquet with Gzip compression
    selected_apat_df.write \
        .option("compression", "gzip") \
        .parquet(output_apat63_99_path)

    spark.stop()


if __name__ == "__main__":
    main(sys.argv)