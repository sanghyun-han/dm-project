import logging
import numpy as np
import pandas as pd
import itertools

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

# for combination
import itertools

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
COL_COMB_RATING="weighted"

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
    def __init__(self, data_size, types, alpha):
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
            
            # comb
            avg = self.train.select(F.avg((F.col(COL_RATING)))).first()[0]
            self.train = self.train.withColumn(COL_COMB_RATING, (alpha*F.col(COL_RATING) + (1-alpha)*avg))
            
            print("train count", self.train.select(COL_RATING).count())
            print("comb count", self.train.select(COL_COMB_RATING).count())
            # get max value
            train_max = self.train.select(F.max((F.col(COL_RATING)))).first()[0]
            test_max = self.test.select(F.max((F.col(COL_RATING)))).first()[0]
            comb_max = self.train.select(F.max((F.col(COL_COMB_RATING)))).first()[0]
            
            # normalize 0~1
            # self.train = self.train.withColumn(COL_RATING, (F.col(COL_RATING) / train_max))
            # self.train = self.train.withColumn(COL_COMB_RATING, (F.col(COL_COMB_RATING) / comb_max))
            # self.test = self.test.withColumn(COL_RATING, (F.col(COL_RATING) / test_max))
            
            print("avg, test_max, train_max, comb_max", avg, test_max, train_max, comb_max)
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
            
        print("ALS Took {} seconds for training.".format(train_time.interval))
        
        return self.model
    def get_output(self, train, user_item, TOP_K):
        with Timer() as infer_time:
            dfs_pred = self.model.transform(user_item)
            # print("# pred: ", dfs_pred.count())
            # print("# train: ", train.count())
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
            # print(dfs_pred_exclude_train)
            # print("# exclude: ", dfs_pred_exclude_train.count())
            top_all = dfs_pred_exclude_train.filter(F.col("train.Rating").isNull())
            top_all = top_all.select(COL_USER, COL_ITEM, "prediction", COL_RATING)
            # print(top_all)
            # print("top all: ", top_all.count())
                
            window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
            
            # inter = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K)
            # print("inter: ", inter)
            # inter.show(30, False)
            
            top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")
            
            
            print("ALSCOUNT", top_k_reco.count())
            print(top_k_reco)
            # print("top_k_reco: ", top_k_reco.count())
        print("ALS Took {} seconds for inference.".format(infer_time.interval))
        
        return top_all, top_k_reco
    
    def get_comb_output(self, top_all, top_k_reco, TOP_K, COMB_LEFT, COMB_RIGHT):
        window = Window.partitionBy(COL_USER).orderBy(F.rand())
        # window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
        # window_1 = Window.partitionBy(COL_USER).orderBy(F.col("group_index").desc())
        avg_col = top_all.select(F.avg(F.col("prediction"))).first()[0]

        new_top_all = top_all.withColumn("prediction", F.col("prediction") + avg_col)
        # new_age_df.limit(5).show()
        
        # top_all = top_all.withColumn("prediction", F.col("prediction") - F.rand()/F.avg("prediction"))
        top_k_reco = top_k_reco.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")
        print(top_k_reco.count())
        # tmp_0 = top_all.withColumn("group_index", F.row_number().over(window) / int(F.max(F.row_number().over(window)) / 3))
        # tmp_1 = tmp_0.withColumn("group", F.row_number().over(window_1))
        # print("tmp 0: ", tmp_0)
        # tmp_0.show(30,False)
        # print("tmp 1: ", tmp_1)
        # tmp_1.show(30,False)
        # comb = asdf
        return new_top_all, top_k_reco
class COMB_MODEL():
    def __init__(self, rank):
        header = {
        "userCol": COL_USER,
        "itemCol": COL_ITEM,
        "ratingCol": COL_COMB_RATING,
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
            
        print("COMB Took {} seconds for training.".format(train_time.interval))
        
        return self.model
    def get_output(self, train, user_item, TOP_K, comb_r):
        with Timer() as infer_time:
            dfs_pred = self.model.transform(user_item)
            
            dfs_pred_exclude_train = train.alias("train").join(
                dfs_pred.alias("pred"),
                [COL_USER, COL_ITEM],
                how='outer'
            )
            
            top_all = dfs_pred_exclude_train.filter(F.col("train.weighted").isNull())
            top_all = top_all.select(COL_USER, COL_ITEM, "prediction", COL_COMB_RATING)
            # top_comb = top_all.withColumn("COMB", F.col)
            # F.locate
            # F.
            
            # self.train = self.train.withColumn(COL_RATING, (F.col(COL_RATING) / train_max))
            
            window_asc = Window.partitionBy(COL_USER).orderBy(F.col("prediction").asc())
            window_desc = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
            window = Window.partitionBy("asc_user").orderBy(F.rand())
            
            top_k_desc = top_all.select("*", F.row_number().over(window_desc).alias("rank")).filter(F.col("rank") <= 20).drop("rank").drop("weighted")
            top_k_asc = top_k_desc.select("*", F.row_number().over(window_asc).alias("rank")).drop("rank").drop("weighted")
            
            # join with combination
            top_k_asc = top_k_asc.selectExpr("UserId as asc_user", "MovieId as asc_item", "prediction as asc_pred")
            top_k_desc = top_k_desc.selectExpr("UserId as desc_user", "MovieId as desc_item", "prediction as desc_pred")
            
            cond = [top_k_asc["asc_user"] == top_k_desc["desc_user"], top_k_asc["asc_item"] != top_k_desc["desc_item"]]
            joined = top_k_asc.join(
                top_k_desc,
                cond,
                how='outer'
            )
            
            # print(joined.show(60))
            
            joined = joined.withColumn("COMB", ((F.col("asc_pred") + F.col("desc_pred"))/2 ))
            # print(joined.show(60))
            joined = joined.dropDuplicates(["COMB"])
            # print(joined.show(60))
            
            # joined_k = joined.withColumn("asc_user", F.col("asc.userId")).withColumn("desc_user", F.col("desc.userId")).drop("asc.userId").drop("desc.userId")
            # joined_k = joined.withColumn("asc_movieId", F.col("asc.movieId")).withColumn("desc_movieId", F.col("desc.movieId")).drop("asc.movieId").drop("desc.movieId")
            # joined_k = joined.withColumn("asc_pred", F.col("asc.prediction")).withColumn("desc_pred", F.col("desc.prediction")).drop("asc.prediction").drop("desc.prediction")
            
            # asc = top_all = top_all.select(COL_USER, COL_ITEM, "prediction", COL_RATING)
            col_names = ['desc_user', 'desc_item', 'desc_pred']
            joined_k = joined.select(*col_names, F.row_number().over(window).alias("rank")).filter(F.col("rank") <= 10).drop("rank")
            joined_k = joined_k.selectExpr("desc_user as UserId", "desc_item as MovieId", "desc_pred as prediction")
            # joined_k = joined_k.dropDuplicates([])
            # print(joined_k.show(60))
            
            # # regenerate recommendations
            # cols = [COL_USER, COL_ITEM, "prediction"]
            # data = []
            
            # # set_item = set()
            # # set_mv = joined_k.select("asc.userId")
            # iter = joined_k.collect()
            # print(iter[0])
            # print(len(iter))
            # print(joined_k.count())
            # idx = 0
            # name = ""
            # while(idx < joined_k.count()):
            #     if idx % 10 == 0 : print("iter: ", idx, len(data), name)
            #     if (len(data) % TOP_K == 0) and name == iter[idx]["asc_user"]:
            #         idx += 1
            #         continue
                
            #     name = iter[idx]["asc_user"]
                
            #     if not [iter[idx]["asc_user"], iter[idx]["asc_item"], iter[idx]["asc_pred"]] in data:
            #         data.append([iter[idx]["asc_user"], iter[idx]["asc_item"], iter[idx]["asc_pred"]])
            #     if not [iter[idx]["desc_user"], iter[idx]["desc_item"], iter[idx]["desc_pred"]] in data:
            #         data.append([iter[idx]["desc_user"], iter[idx]["desc_item"], iter[idx]["desc_pred"]])
                
            #     idx += 1
                
            # top_k_reco = spark.createDataFrame(data, cols)
            # print(top_k_reco.show(60))
            # # print(joined_k["asc.userId"])
            # top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")
            # print("COMBCOUNT", top_k_reco.count())
            # print(top_k_reco)
            # # comb_window = Window.partitionBy(COL_USER).orderBy(F.rand())
            
            # # print(top_k_reco)
            # # print("top_k_reco: ", top_k_reco.count())
        print("COMB Took {} seconds for inference.".format(infer_time.interval))
        
        return top_all, joined_k
    
    def get_comb_output(self, top_all, top_k_reco, TOP_K, user_item, COMB_LEFT, COMB_RIGHT):
        window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
        # window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
        # window_1 = Window.partitionBy(COL_USER).orderBy(F.col("group_index").desc())
        avg_col = top_all.select(F.avg(F.col("prediction"))).first()[0]
        dfs_pred = self.model.transform(user_item)
        
        # dfs_pred_exclude_train = train.alias("train").join(
        #     dfs_pred.alias("pred"),
        #     [COL_USER, COL_ITEM],
        #     how='outer'
        # )
        
        # top_all = dfs_pred_exclude_train.filter(F.col("train.Rating").isNull())
        top_all = top_all.select(COL_USER, COL_ITEM, "prediction", COL_RATING)
        
        
        new_top_all = top_all.withColumn("prediction", F.col("prediction") + avg_col)
        # new_age_df.limit(5).show()
        
        # top_all = top_all.withColumn("prediction", F.col("prediction") - F.rand()/F.avg("prediction"))
        top_k_reco = top_k_reco.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")
        print(top_k_reco)
        
        # tmp_0 = top_all.withColumn("group_index", F.row_number().over(window) / int(F.max(F.row_number().over(window)) / 3))
        # tmp_1 = tmp_0.withColumn("group", F.row_number().over(window_1))
        # print("tmp 0: ", tmp_0)
        # tmp_0.show(30,False)
        # print("tmp 1: ", tmp_1)
        # tmp_1.show(30,False)
        # comb = asdf
        return new_top_all, top_k_reco
        
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
    
    # def get_comb_output(self, train, user_item, TOP_K, COMB_LEFT, COMB_RIGHT):
        
    
class Evaluation():
    def __init__(self, train, test, top_k_reco, top_all, TOP_K, pred_columns):
        diversity_eval = SparkDiversityEvaluation(
            train_df = train, 
            reco_df = top_k_reco,
            col_user = COL_USER, 
            col_item = COL_ITEM
        )
        
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