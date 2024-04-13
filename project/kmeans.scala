package project

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
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

class Conf_KMeans(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output,k)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val k = opt[String](descr = "Number of clusters",required = true)
  //  val shuffle = toggle(default = Some(false))
  //  val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}

object kmeans {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new Conf_KMeans(argv)

    log.info("Input: " + args.input())
    log.info("Output path: " + args.output())
    log.info("Number of clusters: " + args.k())

    //    log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("KMeans"+args.input())
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val sparkSession = SparkSession.builder.getOrCreate
    //    import sparkSession.implicits._
    //    val df = sparkSession.read.csv(args.input())

    val csvData = sparkSession.read
      .option("header","false")
      .option("inferSchema", "true")
      .csv(args.input())

    val assembler = new VectorAssembler()
      .setInputCols(csvData.columns.filter(_ != "label"))
      .setOutputCol("features")

    val df = assembler.transform(csvData)
      .select("features")


    val KMeans = new KMeans().setK(args.k().toInt).setSeed(1L)
    val model = KMeans.fit(df)

    // Make predictions
    val predictions = model.transform(df)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

//    predictions.coalesce(1)
//      .write
//      .mode(SaveMode.Overwrite) // Overwrite or append to the file
//      .option("header", "true") // Write the column names in the first row
//      .csv(args.output())

    predictions.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .json(args.output())



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