from pyspark import SparkConf, SparkContext
import sys

def parse_line(line):
    parts = line.split(',')
    #offset: 0 = PATENT, 1 = GYEAR, 4 = COUNTRY
    country = parts[4].replace('"', '')
    return (country, (parts[1], parts[0]))

def main(args):
    #spark-submit --master yarn --num-executors 8 --driver-memory 4g --queue urgent p3.py /user/curso370/data/apat63_99.txt /user/curso370/output/p3_output_02
    #hdfs dfs -cat /user/curso370/output/p3_output_02/part-00000
    if len(args) != 3:
        print("Usage: pyspark_rdd.py <input_file> <output_file>")
        sys.exit(-1)

    input_file = args[1]
    output_file = args[2]

    # Initialize Spark Context
    conf = SparkConf().setAppName("Practica 3 de Tomas")
    spark = SparkContext(conf=conf)
    spark.setLogLevel("FATAL")

    #to ensure 8 partitions on reading and writing
    lines = spark.textFile(input_file, minPartitions=8)

    #to skip header
    header = lines.first()
    data = lines.filter(lambda line: line != header)

    #Parse lines into RDD
    #tuple: <country, <gyear, patent>>
    country_year_patents = data.map(parse_line)

    #Agg by (COUNTRY,GYEAR) and count(patents)
    country_year_counts = country_year_patents.map(lambda x: ((x[0], x[1][0]), 1)) \
                                              .reduceByKey(lambda a, b: a + b)

    #Reshape to the structure <COUNTRY, <<GYEAR, count>>
    country_year_list = country_year_counts.map(lambda x: (x[0][0], (x[0][1], x[1]))) \
                                           .groupByKey() \
                                           .mapValues(list)

    #Sort the list by GYEAR for each COUNTRY
    sorted_country_year_list = country_year_list.mapValues(lambda x: sorted(x, key=lambda y: y[0]))

    #Sort by COUNTRY
    sorted_result = sorted_country_year_list.sortByKey()

    #Save to output file in HDFS with 8 partitions
    sorted_result.saveAsTextFile(output_file)

    spark.stop()

if __name__ == "__main__":
    main(sys.argv)
