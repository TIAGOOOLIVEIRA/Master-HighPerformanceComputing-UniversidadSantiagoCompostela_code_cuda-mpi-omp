from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg as _avg, max as _max
from pyspark.sql import functions as F

import sys

#module load anaconda3
#spark-submit --master yarn --num-executors 8 --driver-memory 4g --queue urgent p2.py /user/curso370/output/cite75_99.parquet /user/curso370/output/apat63_99.parquet /user/curso370/data/country_codes.txt /user/curso370/output/p2_out_01.parquet
#spark-submit --master yarn --num-executors 8 --driver-memory 4g --queue urgent --files /home/ulc/cursos/curso370/tarea4_tema9_spark_mapreduce/patentes/country_codes.txt p2_2.py /user/curso370/output/cite75_99.parquet /user/curso370/output/apat63_99.parquet /user/curso370/output/p2_out_05.parquet
#hdfs dfs -ls /user/curso370/output/p2_out_01.parquet
#hdfs dfs -ls /user/curso370/output/p2_out_05.parquet

def main(args):
    if len(args) != 4:
        print("Usar: p2.py <dirNcitas> <dirInfo> <output_path>")
        sys.exit(-1)

    dirNcitas = args[1]
    dirInfo = args[2]
    #country_codes_path = args[3]
    output_path = args[3]

    country_codes_path = "/home/ulc/cursos/curso370/tarea4_tema9_spark_mapreduce/patentes/country_codes.txt"

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Practica 2 de Tomas") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("FATAL")

    cite_df = spark.read.parquet(dirNcitas)
    info_df = spark.read.parquet(dirInfo)

    #country_codes_df = spark.read.option("header", False).option("delimiter", "\t").csv(country_codes_path).toDF("country_code", "country_name")

    countries_dict = {}
    with open(country_codes_path, "r") as f:
        for line in f:
            code, name = line.strip().split("\t")
            countries_dict[code] = name


    #Convert DataFrame to a dictionary and broadcast it
    #country_codes_broadcast = spark.sparkContext.broadcast(dict(country_codes_df.collect()))
    country_codes_broadcast = spark.sparkContext.broadcast(countries_dict)

    #UDF to map countryCode->countryNames, leveraging the broadcasted dictionary
    def get_country_name(country_code):
        return country_codes_broadcast.value.get(country_code, country_code)

    #Register the UDF
    spark.udf.register("get_country_name", get_country_name)

    # Mapping UDF to country codes to country names
    info_df = info_df.withColumn("Pais", F.expr("get_country_name(Pais)"))

    # Join on NPatente key
    joined_df = info_df.join(cite_df, info_df.NPatente == cite_df.NPatente, "inner") \
                       .select(cite_df.NPatente, info_df.Pais, info_df.Anho, cite_df.ncitas)

    # Group by Pais and Anho, and aggregate
    result_df = joined_df.groupBy("Pais", "Anho") \
                         .agg(F.count("NPatente").alias("NumPatentes"),
                              _sum("ncitas").alias("TotalCitas"),
                              _avg("ncitas").alias("MediaCitas"),
                              _max("ncitas").alias("MaxCitas"))

    # Write to a CSV file
    result_df.write.option("header", True).csv(output_path, mode="overwrite")

    spark.stop()

if __name__ == "__main__":    
    main(sys.argv)
