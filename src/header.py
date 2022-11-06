import logging
import numpy as np
import pandas as pd

from datasets import movielens
from datasets.python_splitters import python_stratified_split
from utils.python_utils import binarize
from utils.timer import Timer
from datasets import movielens
from datasets.python_splitters import python_stratified_split
from evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from models.sar import SAR
import sys

import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StructType, StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, CountVectorizer, VectorAssembler

from utils.timer import Timer
from datasets import movielens
from datasets.spark_splitters import spark_random_split
from evaluation.spark_evaluation import SparkRankingEvaluation, SparkDiversityEvaluation
from utils.spark_utils import start_or_get_spark

from pyspark.sql.window import Window
import pyspark.sql.functions as F

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

# pyspark.sql.analyzer.failAmbiguousSelfJoin = False
spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.crossJoin.enabled", "true")
# spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

COL_USER="UserId"
COL_ITEM="MovieId"
COL_RATING="Rating"
COL_TITLE="Title"
COL_GENRE="Genre"

def get_ranking_results(ranking_eval):
    metrics = {
        "Precision@k": ranking_eval.precision_at_k(),
        "Recall@k": ranking_eval.recall_at_k(),
        "NDCG@k": ranking_eval.ndcg_at_k(),
        "Mean average precision": ranking_eval.map_at_k()
      
    }
    return metrics   

def get_diversity_results(diversity_eval):
    metrics = {
        "catalog_coverage":diversity_eval.catalog_coverage(),
        "distributional_coverage":diversity_eval.distributional_coverage(), 
        "novelty": diversity_eval.novelty(), 
        "diversity": diversity_eval.diversity(), 
        "serendipity": diversity_eval.serendipity()
    }
    return metrics 

def generate_summary(data, algo, k, ranking_metrics, diversity_metrics):
    summary = {"Data": data, "Algo": algo, "K": k}

    if ranking_metrics is None:
        ranking_metrics = {           
            "Precision@k": np.nan,
            "Recall@k": np.nan,            
            "nDCG@k": np.nan,
            "MAP": np.nan,
        }
    summary.update(ranking_metrics)
    summary.update(diversity_metrics)
    return summary

class DataLoader():
    def __init__(self, data_size, types):
        if types == "movielens":
            data = movielens.load_pandas_df(size=data_size)
            # Convert the float precision to 32-bit in order to reduce memory consumption
            data['rating'] = data['rating'].astype(np.float32)
            data.head()
            self.train, self.test = python_stratified_split(data, ratio=0.75, col_user='userID', col_item='itemID', seed=42)
            print("""
            Train:
            Total Ratings: {train_total}
            Unique Users: {train_users}
            Unique Items: {train_items}

            Test:
            Total Ratings: {test_total}
            Unique Users: {test_users}
            Unique Items: {test_items}
            """.format(
                train_total=len(self.train),
                train_users=len(self.train['userID'].unique()),
                train_items=len(self.train['itemID'].unique()),
                test_total=len(self.test),
                test_users=len(self.test['userID'].unique()),
                test_items=len(self.test['itemID'].unique()),
            ))
        if types == "spark":
            schema = StructType(
                (
                    StructField(COL_USER, IntegerType()),
                    StructField(COL_ITEM, IntegerType()),
                    StructField(COL_RATING, FloatType()),
                    StructField("Timestamp", LongType()),
                )
            )
            data = movielens.load_spark_df(spark, size=data_size, schema=schema, title_col=COL_TITLE, genres_col=COL_GENRE)
            print("movielens data: ", data)
            data = data.select(COL_USER, COL_ITEM, COL_RATING)
            self.train, self.test = spark_random_split(data, ratio=0.75, seed=123)
            print ("N train", self.train.cache().count())
            print ("trainset: ", self.train)
            print ("N test", self.test.cache().count())
            print ("testset: ", self.test)
            users = self.train.select(COL_USER).distinct()
            items = self.train.select(COL_ITEM).distinct()
            self.user_item = users.crossJoin(items)
            print("user item: ", self.user_item)
        else:
            print ("Type error\n")
class SAR_MODEL():
    def __init___(self):
        self.model = SAR(
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_timestamp="timestamp",
            similarity_type="jaccard", 
            time_decay_coefficient=30, 
            timedecay_formula=True,
            normalize=True
        )
class ALS_MODEL():
    def __init__(self):
        header = {
        "userCol": COL_USER,
        "itemCol": COL_ITEM,
        "ratingCol": COL_RATING,
        }
        self.als = ALS(
            rank=10,
            maxIter=15,
            implicitPrefs=False,
            regParam=0.05,
            coldStartStrategy='drop',
            nonnegative=False,
            seed=42,
            **header
        )
        self.model = None
    
    def train(self, train):
        with Timer() as train_time:
            self.model = self.als.fit(train)
            
        print("Took {} seconds for training.".format(train_time.interval))
        
        return self.model
    def get_output(self, train, user_item, TOP_K):
        dfs_pred = self.model.transform(user_item)
        print("# pred: ", dfs_pred.count())
        print("# train: ", train.count())
        # Remove seen items.
        # cond = [
        #     # ((right_side.lower_street_number.isNotNull())
        # ]
        # dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        #     train.alias("train"),
        #     [COL_USER, COL_ITEM],
        #     how='outer'
        # )
        dfs_pred_exclude_train = train.alias("train").join(
            dfs_pred.alias("pred"),
            [COL_USER, COL_ITEM],
            how='outer'
        )
        print(dfs_pred_exclude_train)
        # print(dfs_pred_exclude_train["train.Rating"])
        # print(dfs_pred_exclude_train["pred."+ COL_ITEM])
        # print(dfs_pred_exclude_train["pred." + COL_USER])
        # print(dfs_pred_exclude_train["pred." + "prediction"])
        # print(dfs_pred_exclude_train["train.Rating"])
        print("# exclude: ", dfs_pred_exclude_train.count())
        top_all = dfs_pred_exclude_train.filter(F.col("train.Rating").isNull())
        top_all = top_all.select(COL_USER, COL_ITEM, "prediction", COL_RATING)
        print(top_all)
        print("top all: ", top_all.count())
            
        window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
        top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")
        
        print(top_k_reco)
        print("top_k_reco: ", top_k_reco.count())
        
        return top_all, top_k_reco
        # als_eval = Evaluation(movielens.train, movielens.test, top_k_reco, top_all, TOP_K)
class RANDOM_MODEL():
    def __init__(self):
        self.model = Window.partitionBy(COL_USER).orderBy(F.rand())
    
    def get_output(self, train, user_item, TOP_K):
        pred_df = (
            train
            # join training data with all possible user-item pairs (seen in training)
            .join(user_item,
                    on=[COL_USER, COL_ITEM],
                    how="right"
            )
            # get user-item pairs that were not seen in the training data
            .filter(F.col(COL_RATING).isNull())
            # count items for each user (randomly sorting them)
            .withColumn("score", F.row_number().over(self.model))
            # get the top k items per user
            .filter(F.col("score") <= TOP_K)
            .drop(COL_RATING)
        )
        return pred_df
    
class Evaluation():
    def __init__(self, train, test, top_k_reco, top_all, TOP_K, pred_columns):
        diversity_eval = SparkDiversityEvaluation(
            train_df = train, 
            reco_df = top_k_reco,
            col_user = COL_USER, 
            col_item = COL_ITEM
        )
        print(test)
        print(test.columns)
        if COL_RATING not in test.columns:
            raise ValueError("Schema of rating_true not valid. Missing Rating Col")
        ranking_eval = SparkRankingEvaluation(
            rating_true=test,
            rating_pred=top_all, 
            k = TOP_K, 
            col_user=COL_USER, 
            col_item=COL_ITEM,
            col_rating=COL_RATING, 
            col_prediction=pred_columns,
            relevancy_method="top_k"
        )
        self.ranking_metrics = get_ranking_results(ranking_eval)
        self.diversity_metrics = get_diversity_results(diversity_eval)
    
    def get_metrics(self):
        return self.ranking_metrics, self.diversity_metrics
    
    def get_results(self, data_size, TOP_K, name):
        results = generate_summary(data_size, name, TOP_K, self.ranking_metrics, self.diversity_metrics)
        return results