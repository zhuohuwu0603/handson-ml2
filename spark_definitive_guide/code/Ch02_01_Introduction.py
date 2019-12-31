import inspect

def Ch02_01_Introduction():
  print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName('Spark-Definitive-Guide').getOrCreate()
  BASEPATH = "/Users/zhuohuawu/Documents/data/spark-definitive-guide"
  # BASEPATH = "spark_definitive_guide/"
  path = BASEPATH + "/data/flight-data/csv/2015-summary.csv"
  print("path={}".format(path))

  myRange = spark.range(1000).toDF("number")

  divisBy2 = myRange.where("number % 2 = 0")

  flightData2015 = spark \
    .read \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .csv(path)

  flightData2015.createOrReplaceTempView("flight_data_2015")

  sqlWay = spark.sql("""
  SELECT DEST_COUNTRY_NAME, count(1)
  FROM flight_data_2015
  GROUP BY DEST_COUNTRY_NAME
  """)

  dataFrameWay = flightData2015 \
    .groupBy("DEST_COUNTRY_NAME") \
    .count()

  sqlWay.explain()
  dataFrameWay.explain()

  from pyspark.sql.functions import max

  flightData2015.select(max("count")).take(1)

  maxSql = spark.sql("""
  SELECT DEST_COUNTRY_NAME, sum(count) as destination_total
  FROM flight_data_2015
  GROUP BY DEST_COUNTRY_NAME
  ORDER BY sum(count) DESC
  LIMIT 5
  """)

  maxSql.show()

  from pyspark.sql.functions import desc

  flightData2015 \
    .groupBy("DEST_COUNTRY_NAME") \
    .sum("count") \
    .withColumnRenamed("sum(count)", "destination_total") \
    .sort(desc("destination_total")) \
    .limit(5) \
    .show()

  flightData2015 \
    .groupBy("DEST_COUNTRY_NAME") \
    .sum("count") \
    .withColumnRenamed("sum(count)", "destination_total") \
    .sort(desc("destination_total")) \
    .limit(5) \
    .explain()

  print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))

def test():
  print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

  print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
  Ch02_01_Introduction()
  test()


