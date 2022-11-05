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

COL_USER="UserId"
COL_ITEM="MovieId"
COL_RATING="Rating"
COL_TITLE="Title"
COL_GENRE="Genre"

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
            self.spark = start_or_get_spark("ALS PySpark", memory="16g")
            self.spark.conf.set("spark.sql.crossJoin.enabled", "true")
            schema = StructType(
                (
                    StructField(COL_USER, IntegerType()),
                    StructField(COL_ITEM, IntegerType()),
                    StructField(COL_RATING, FloatType()),
                    StructField("Timestamp", LongType()),
                )
            )
            data = movielens.load_spark_df(self.spark, size=data_size, schema=schema, title_col=COL_TITLE, genres_col=COL_GENRE)
            self.train, self.test = spark_random_split(data.select(COL_USER, COL_ITEM, COL_RATING), ratio=0.75, seed=123)
            print ("N train", self.train.cache().count())
            print ("N test", self.test.cache().count())
class SAR():
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
class Evaluation():
    def __init__(self):
        