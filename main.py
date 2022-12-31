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

    return user_matrix


def als_step(train_matrix, theta, beta, variable, learning_rate):
    '''
    A single step of the Alternating Least Squares algorithm
    :param train_matrix: Matrix we desire
    :param variable_matrix: Matrix we're changing
    :param constant_matrix: Matrix we're assuming to be constants
    :param variable: States which matrix is being altered
    :param learning_rate: Learning rate
    :return: The changed variable_matrix
    '''

    if variable == "theta":
        return np.subtract(theta, np.multiply(learning_rate, np.subtract(train_matrix, np.dot(theta, beta.T)))@beta)
    else:
        return np.subtract(beta, np.multiply(learning_rate, np.subtract(train_matrix, np.dot(theta, beta.T))).T@theta)


def calculate_loss(train_matrix, pred):
    '''
    Calculates total loss.
    :param train_matrix:
    :param pred:
    :return: Total summed loss
    '''
    mask = train_matrix != 0
    return sum(np.abs(train_matrix[mask] - pred[mask]).reshape(-1))

def main():
    # Hyperparameters
    k = 3
    iterations = 100
    learning_rate = 1

    # Data
    train_matrix = RatingMatrixDataset(["./ml-100k/u1.base"])
    validation_matrix = RatingMatrixDataset(["./ml-100k/u2.base"])

    # Initialize both matrices
    theta = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(943, k))
    beta = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(1682, k))

    for _ in range(iterations):
        theta = als_step(train_matrix, theta, beta, "theta", learning_rate)
        beta = als_step(train_matrix, theta, beta, "beta", learning_rate)
        pred = np.dot(theta, beta.T)
        print("Training Loss" + str(calculate_loss(train_matrix, pred)))
        print("Validation Loss" + str(calculate_loss(validation_matrix, pred)))


main()
