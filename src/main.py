
import argparse
import os
import pandas as pd

from header import MovieLens

# parse argument
parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--data-path', type=str, help='data path of movielens')
parser.add_argument('--top-k', type=int, help='the number of k for recommendations')
parser.add_argument('--comb', type=int, help='the number of combination for grouping')
parser.add_argument('--data-size', type=str, help='the data size of movielens')


# movies = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'ml-25m/movies.csv'))
# movies.head()

def main():
    args = parser.parse_args()
    movielens = MovieLens(args.data_size)
    # movies = ml.get_movie()
    # rates = ml.get_rate()
    
    # print(movies)
    # print(rates)
    
if __name__ == "__main__":
    main()