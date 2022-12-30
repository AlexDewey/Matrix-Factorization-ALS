import math

import pandas as pd
import numpy as np
import torch


def RatingMatrixDataset(destinations):
    '''
    Grabs dataset location and creates user-item matrix.
    :param destinations: Array of file locations for movie-lens
    :return: user-item matrix and binary filter
    '''
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = None
    for destination in destinations:
        if data is None:
            data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
        else:
            new_data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
            data = pd.concat([data, new_data], axis=0)

    test_matrix = torch.zeros(943, 1682)

    user_matrix = torch.zeros(943, 1682)
    for index, row in data.iterrows():
        user_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']

    filter = user_matrix > 0

    return user_matrix, filter


def main():
    # Hyperparameters
    k = 3

    # Data
    train_matrix, train_filter = RatingMatrixDataset(["./ml-100k/u1.base"])
    valid_set, valid_filter = RatingMatrixDataset(["./ml-100k/u2.base"])

    # Alternating Least Squares (ALS) Setup
    theta = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(943, k))
    beta = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(1682, k))


main()
