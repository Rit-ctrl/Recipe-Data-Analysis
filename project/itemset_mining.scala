package project


import org.apache.spark.ml.fpm.FPGrowth

import org.apache.spark.sql.{ SparkSession,Encoders,SaveMode}

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._

class conf_itemset(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output,minSupport,minConfidence)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val minSupport = opt[String](descr = "minimum support",required = true)
  val minConfidence = opt[String](descr = "minimum Confidence",required = true)
  verify()
}

object itemset_mining {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new conf_itemset(argv)

    log.info("Input: " + args.input())
    log.info("Output path: " + args.output())
    log.info("Min Support: " + args.minSupport())
    log.info("Min Confidence" + args.minConfidence())

    val conf = new SparkConf().setAppName("Itemset mining"+args.input())
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val sparkSession = SparkSession.builder.getOrCreate

//    val csvData = sparkSession.read
//      .option("header","false")
//      .option("inferSchema", "true")
//      .csv(args.input())
//
//    val df = csvData.toDF("items")

    val raw_data = sparkSession.read
      .option("header","true")
      .option("multiline","true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("inferSchema", "true")
      .csv(args.input())
      .rdd

    val NER = raw_data.map(row => row.getString(5))
      .map(row => {row.substring(1,row.length-1).replace("\"","")})
      .map(row => {
        val ingredients = row.split(",") //.filter(_.trim.nonEmpty)
        val unique_ingredients = ingredients.distinct
        unique_ingredients.toSeq})
      .map(Tuple1.apply)
//      .take(10)

    val df = sparkSession.createDataFrame(NER)
      .toDF("items")
//    val assembler = new VectorAssembler()
//      .setInputCols(csvData.columns.filter(_ != "label"))
//      .setOutputCol("items")
//
//    val df = assembler.transform(csvData)
//      .select("items")

    val fpgrowth = new FPGrowth()
      .setItemsCol("items")
      .setMinSupport(args.minSupport().toFloat)
      .setMinConfidence(args.minConfidence().toFloat)

    val model = fpgrowth.fit(df)

    model.freqItemsets.show()
    model.associationRules.show()

    model.associationRules.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .json(args.output()+"associationRules")

    model.freqItemsets.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .json(args.output()+"freqItemsets")


    model.transform(df).show()

    sparkSession.stop()



  }
}