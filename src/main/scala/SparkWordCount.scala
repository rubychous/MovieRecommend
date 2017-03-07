import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark._  

object SparkWordCount { 


def main(args: Array[String]) {
/**
* Parameters to regularize correlation.
*/
val PRIOR_COUNT = 10
val PRIOR_CORRELATION = 0


/**
* Spark programs require a SparkContext to be initialized
*/
val conf = new SparkConf().setAppName("Movie Similarities")
val sc = new SparkContext(conf)

// get movie names keyed on id
val movies = sc.textFile("/data/movie-ratings/movies.dat",20).map(line => {
val fields = line.split("::")
(fields(0).toInt, fields(1))})
val movieNames = movies.collectAsMap()    // get MovieNames,MovieId as key value pairs as ratings data has movieId only

// extract (userid, movieid, rating) from ratings data
val ratings = sc.textFile("/data/movie-ratings/ratings.dat",5).map(line => {
val fields = line.split("::")
(fields(0).toInt, fields(1).toInt, fields(2).toFloat)})

val rtgrp=ratings.groupBy(x => x._2).repartition(1500)//Partition the ratings data after groupBy ,The map functions after group by can perform actions parallely
rtgrp.cache// Cache rtgrp to calculate numRatersPerMovie and ratingsWithSize parallely and effectively avoid repeated reading of data from hdfs 

// get num raters per movie, keyed on movie id
val numRatersPerMovie = rtgrp.map(m => (m._1, m._2.size))

// join ratings with num raters on movie id
val ratingsWithSize = rtgrp.join(numRatersPerMovie).flatMap(joined => {joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))})


// ratingsWithSize now contains the following fields: (user, movie, rating, numRaters).

// dummy copy of ratings for self join
val ratings2 = ratingsWithSize.keyBy(x => x._1)

// join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
val ratingPairs =ratingsWithSize.keyBy(x => x._1).join(ratings2).filter(f => f._2._1._2 < f._2._2._2)
rtgrp.unpersist() //remove rtgrp from cache as it is not needed for future calculations

// compute raw inputs to similarity metrics for each movie pair
val vectorCalcs = ratingPairs.map(data => {
val key = (data._2._1._2, data._2._2._2)
val stats =
  (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
    data._2._1._3,                // rating movie 1
    data._2._2._3,                // rating movie 2
    math.pow(data._2._1._3, 2),   // square of rating movie 1
    math.pow(data._2._2._3, 2),   // square of rating movie 2
    data._2._1._4,                // number of raters movie 1
    data._2._2._4)                // number of raters movie 2
(key, stats)
}).groupByKey().map(data => {
val key = data._1
val vals = data._2
val size = vals.size
val dotProduct = vals.map(f => f._1).sum
val ratingSum = vals.map(f => f._2).sum
val rating2Sum = vals.map(f => f._3).sum
val ratingSq = vals.map(f => f._4).sum
val rating2Sq = vals.map(f => f._5).sum
val numRaters = vals.map(f => f._6).max
val numRaters2 = vals.map(f => f._7).max
(key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
})

//Similarity Metrics
/**
* The correlation between two vectors A, B is
*   cov(A, B) / (stdDev(A) * stdDev(B))
*
* This is equivalent to
*   [n * dotProduct(A, B) - sum(A) * sum(B)] /
*     sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }
*/
def correlation(size : Double, dotProduct : Double, ratingSum : Double,
          rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double) = {

val numerator = size * dotProduct - ratingSum * rating2Sum
val denominator = scala.math.sqrt(size * ratingNormSq - ratingSum * ratingSum) *
scala.math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

numerator / denominator
}

/**
* Regularize correlation by adding virtual pseudocounts over a prior:
*   RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
* where w = # actualPairs / (# actualPairs + # virtualPairs).
*/
def regularizedCorrelation(size : Double, dotProduct : Double, ratingSum : Double,
                     rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double,
                     virtualCount : Double, priorCorrelation : Double) = {

val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
val w = size / (size + virtualCount)

w * unregularizedCorrelation + (1 - w) * priorCorrelation
}

/**
* The cosine similarity between two vectors A, B is
*   dotProduct(A, B) / (norm(A) * norm(B))
*/
def cosineSimilarity(dotProduct : Double, ratingNorm : Double, rating2Norm : Double) = {
dotProduct / (ratingNorm * rating2Norm)
}

/**
* The Jaccard Similarity between two sets A, B is
*   |Intersection(A, B)| / |Union(A, B)|
*/
def jaccardSimilarity(usersInCommon : Double, totalUsers1 : Double, totalUsers2 : Double) = {
val union = totalUsers1 + totalUsers2 - usersInCommon
usersInCommon / union}

// compute similarity metrics for each movie pair
val similarities =vectorCalcs.map(fields => 
{val key = fields._1 
val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,ratingNormSq, rating2NormSq, 10, 0)
val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

(key, (corr, regCorr, cosSim, jaccard))
})


similarities.cache
val data=sc.textFile("/user/gpg31/input/prasanna.txt").map(line => { val fields =line.split(",") //read new users movie history
(fields(1), fields(2).toFloat)}).collect()

val liking=data.filter(x=>x._2>3).map(m=>m._1)//get users movies which are rated above 3


val likingsimilarity = similarities.filter(m => {liking.contains(movieNames(m._1._1))})//find movie pairs which has the movies that user liked


similarities.unpersist()//remove the similarities calculated from the cache

//Map all the similarity metrics to movie pairs
val temp= likingsimilarity.map(v => {
val m1 = v._1._1
val m2 = v._1._2
val corr = v._2._1
val rcorr = v._2._2
val cos = v._2._3
val j = v._2._4
(movieNames(m1), movieNames(m2), corr, rcorr, cos, j)
}).filter(e => !(e._4 equals Double.NaN)) //Remove NaN values


temp.cache
//retreive top results from all the similarity metrics used
val relative=temp.sortBy(elem => -elem._4).take(4)
val cosine=temp.sortBy(elem => -elem._5).take(3)
val jacard=temp.sortBy(elem => -elem._6).take(3)


val result=(relative++cosine++jacard++relativediff).toList.distinct//concatenate the results

temp.unpersist()


sc.makeRDD(result).saveAsTextFile("/user/gpg31/endresult/")//save result in hdfs
}

}


