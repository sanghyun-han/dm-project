
import argparse
import os
import pandas as pd
from utils.timer import Timer

from header import DataLoader, ALS_MODEL, SAR_MODEL, RANDOM_MODEL, Evaluation
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# user, item column names
COL_USER="UserId"
COL_ITEM="MovieId"
COL_RATING="Rating"
COL_TITLE="Title"
COL_GENRE="Genre"

# parse argument
parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--data-path', type=str, help='data path of movielens')
parser.add_argument('--top-k', type=int, help='the number of k for recommendations')
parser.add_argument('--comb', type=int, help='the number of combination for grouping')
parser.add_argument('--data-size', type=str, help='the data size of movielens')
parser.add_argument('--comb-r', type=int, help='right combination')
parser.add_argument('--comb-l', type=int, help='left combination')


# movies = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'ml-25m/movies.csv'))
# movies.head()

def main():
    # argument & model & data loader
    args = parser.parse_args()
    movielens = DataLoader(args.data_size, types="spark")
    TOP_K = args.top_k
    
    # als model
    als = ALS_MODEL()
    als.train(movielens.train)
    top_all, top_k_reco = als.get_output(movielens.train, movielens.user_item, TOP_K)
    
    # random model
    rnd = RANDOM_MODEL()
    pred = rnd.get_output(movielens.train, movielens.user_item, TOP_K)
    
    # combination of the outputs
    TOP_COMB = 20
    comb = ALS_MODEL()
    comb.train(movielens.train)
    comb_top_all, comb_top_k_reco = comb.get_output(movielens.train, movielens.user_item, TOP_COMB)
    comb.get_comb_output(comb_top_all, comb_top_k_reco, TOP_K, args.comb_l, args.comb_r)
    # rnd.get_comb_output(movielens.train, movielens.user_item, TOP_K, args.comb_l, args.comb_r)
    
    # save the output
    top_all.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("../data/top_all")
    top_k_reco.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("../data/top_k")
    pred.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("../data/pred")
    
    # evaluation 
    als_eval = Evaluation(movielens.train, movielens.test, top_k_reco, top_all, TOP_K, "prediction")
    rnd_eval = Evaluation(movielens.train, movielens.test, pred, pred, TOP_K, "score")
    comb_eval = Evaluation(movielens.train, movielens.test, comb_top_k_reco, comb_top_all, TOP_K, "prediction")
    
    als_results = als_eval.get_results(args.data_size, TOP_K, "als")
    rnd_results = rnd_eval.get_results(args.data_size, TOP_K, "random")
    comb_results = comb_eval.get_results(args.data_size, TOP_K, "comb")
    
    # print results
    cols = ["Data", "Algo", "K", "Precision@k", "Recall@k", "NDCG@k", "Mean average precision","catalog_coverage", "distributional_coverage","novelty", "diversity", "serendipity" ]
    
    df_results = pd.DataFrame(columns=cols)
    df_results.loc[1] = als_results 
    df_results.loc[2] = rnd_results
    df_results.loc[3] = comb_results
    
    print(df_results)
    df_results.to_csv("../data/results" + args.data_size + ".csv")
if __name__ == "__main__":
    main()