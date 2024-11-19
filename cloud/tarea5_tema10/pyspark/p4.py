from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, rank
from pyspark.sql.window import Window
import sys

def main(args):
    #spark-submit --master yarn  --num-executors 4 --driver-memory 4g p4.py /user/curso370/output/cite75_99.parquet /user/curso370/output/apat63_99.parquet FR,ES /user/curso370/output/p4_output_01
    #hdfs dfs -ls /user/curso370/output/p4_output_01
    if len(args) != 5:
        print("Usar: p4.py input_citas input_info countries output_path")
        sys.exit(-1)

    input_citas = args[1]
    input_info = args[2]
    countries = args[3]
    output_path = args[4]

    conf = SparkConf().setAppName("Practica 4 de Tomas")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    df_citas = sqlContext.read.parquet(input_citas)
    df_info = sqlContext.read.parquet(input_info)

    #Join dataframes on NPatente
    df_joined = df_info.join(df_citas, "NPatente")

    #Filtering the countries given as input
    countries_list = countries.split(",")
    df_filtered = df_joined.filter(df_joined.Pais.isin(countries_list))

    #leveraging windowing functions to rank the patents by number of citations in each country and year
    window_spec = Window.partitionBy("Pais", "Anho").orderBy(col("ncitas").desc())

    #Adding the rank column
    df_ranked = df_filtered.withColumn("Rango", rank().over(window_spec))

    #Defining the columns to be selected and sorted
    df_result = df_ranked.select("Pais", "Anho", "NPatente", "ncitas", "Rango") \
                         .orderBy("Pais", "Anho", "Rango")

    df_result.write.csv(output_path, header=True, mode="overwrite")

    sc.stop()

if __name__ == "__main__":
    main(sys.argv)
