package project

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.apache.spark.ml.linalg.Vector

import scala.collection.mutable.Map
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
import org.apache.spark.Partitioner

class Conf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output,vectorsize)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val vectorsize = opt[String](descr = "vector size",required = true)
//  val shuffle = toggle(default = Some(false))
  //  val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}

object embedding {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new Conf(argv)

    log.info("Input: " + args.input())
    log.info("Model path: " + args.output())
    log.info("Vector size: "+args.vectorsize())
    //    log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("Clustering"+args.input())
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val sparkSession = SparkSession.builder.getOrCreate
//    import sparkSession.implicits._
//    val df = sparkSession.read.csv(args.input())

    val df = sparkSession.read
      .option("header","true")
      .option("multiline","true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("inferSchema", "true")
      .csv(args.input())
      .rdd

    val NER = df.map(row => row.getString(5))
      .map(row => {row.substring(1,row.length-1).replace("\"","")})
      .map(row => {
        val ingredients = row.split(",")
        val unique_ingredients = ingredients.distinct
        unique_ingredients.toSeq})
      .map(Tuple1.apply)
//      .take(10)

    val documentDF = sparkSession.createDataFrame(NER)
      .toDF("ing")
//      .take(3)
//      .foreach(println)

    val word2Vec = new Word2Vec()
      .setInputCol("ing")
      .setOutputCol("result")
      .setVectorSize(args.vectorsize().toInt)
      .setMinCount(0)

    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)

    result.take(3).foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

//    result.saveasTextFile(args.output())
//      result.select(result.columns(1)).write.mode(SaveMode.Overwrite)
//        .option("header","true")
//        .csv(args.output())

//    result.coalesce(1)
//      .write
//      .text(args.output())
      val vectors = result.coalesce(1)
        .select(result.columns(1))
        .rdd
        .map(row => {
          val str = row(0).toString
          str.substring(1,str.length-1)})
        .saveAsTextFile(args.output())

//        val vector_df = sparkSession.createDataFrame(vectors,
//            StructType(StructField("values", ArrayType(DoubleType), nullable = false) :: Nil))
//
//    vector_df.write
//      .mode("overwrite")
//      .option("header", "true")
//      .csv(args.output())
//        .map()
//        ve.saveAsTextFile(args.output())


//      val vectors = result.collect()
//        .map(row => {
//          row(1)
//        })
//        .map(Tuple1.apply)
//        .toDF("value")
//        .write
//        .option("delimiter", ",")
//      .text(args.output())
//          .take(2)
//          .foreach(println)



    //    println(df.count())

//    val fields = df.map(row => row(1))
//    .take(2).foreach(println)
//    println(df.schema(df.columns(5)).dataType)

//    val fields = df.select(df.columns(5))
//      .map(row => row.getAs[String]("NER"))
////      .map(line => {
////        println(line.getClass)
////        line
////      })
//      .take(2)
//      .foreach(println)



//    df.show()
//    .foreach(println)
      sparkSession.stop()
  }

}