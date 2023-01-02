import math

import pandas as pd
import numpy as np
import torch


def RatingMatrixDataset(destinations):
    """
    Grabs dataset location and creates user-item matrix.
    :param destinations: Array of file locations for movie-lens
    :return: user-item matrix and binary filter
    """
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = None
    for destination in destinations:
        if data is None:
            data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
        else:
            new_data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
            data = pd.concat([data, new_data], axis=0)

    user_matrix = torch.zeros(943, 1682)
    for index, row in data.iterrows():
        user_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']

    from random import randrange
    validation_matrix = torch.zeros(943, 1682)
    for index, row in enumerate(user_matrix):
        indices = torch.nonzero(row)
        index_chosen = randrange(len(indices))
        rating = row[indices[index_chosen]]
        user_matrix[index][indices[index_chosen]] = 0
        validation_matrix[index][indices[index_chosen]] = rating

    return user_matrix, validation_matrix


def als_step(train_matrix, x, y, variable, k):
    """
    ALS Step as defined by x = (sum(y*yT) + reg)^-1 * sum(rating*y)
    or y = (sum(x*xT) + reg)^-1 * sum(rating*x)
    :param train_matrix: True matrix with validation scores taken out.
    :param x: x factor
    :param y: y factor
    :param variable: If either x or y is being updated
    :param k: Factor length
    :return: Updated factor chosen by variable
    """
    if variable == "x":
        for u_index, r_u in enumerate(train_matrix):
            # A and B block calculations
            r_ui_indices = torch.nonzero(r_u)
            A = np.zeros([k, k])
            B = np.zeros([k, 1])
            # Looking at every Rui
            for r_ui_index in r_ui_indices:
                y_i = y.T[:][r_ui_index].reshape([k, 1])
                A = np.add(A, np.dot(y_i, y_i.T))
                B = np.add(B, np.multiply(r_u[r_ui_index], y_i))

            x[u_index] = 


def calculate_loss(train_matrix, pred):
    """
    Calculates total loss.
    :param train_matrix:
    :param pred:
    :return: Total summed loss
    """
    mask = train_matrix != 0
    return sum(np.abs(train_matrix[mask] - pred[mask]).reshape(-1))


def als_matrix_factorization(k, iterations, train_matrix, validation_matrix, show_plot):
    """
    ALS Matrix Factorization that can either show or just write results.
    :param k: K number of factors.
    :param iterations: Number of iterations.
    :param learning_rate: Learning rate, too low may result in error.
    :param train_matrix: The matrix to be learned.
    :param validation_matrix: The matrix to be validated against.
    :param show_plot: Whether to show the plot.
    :return: Returns nothing. Could be set up to return last validation_loss[-1] if desired. Maybe useful for other
    purposes.
    """
    x = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(k, train_matrix.size(dim=0)))
    y = np.random.normal(loc=0, scale=1 / math.sqrt(k), size=(k, train_matrix.size(dim=1)))

    train_loss = list()
    validation_loss = list()
    for _ in range(iterations):
        x = als_step(train_matrix, x, y, "x", k)
        y = als_step(train_matrix, x, y, "y", k)
        pred = np.dot(x, y.T)
        train_loss.append(calculate_loss(train_matrix, pred))
        validation_loss.append(calculate_loss(validation_matrix, pred))

    if show_plot:
        from matplotlib import pyplot as plt
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.show()


def main():
    # Hyperparameters
    k_array = [10]
    iterations = 100

    # Training matrix leaves out validation matrix which is a single rating from each user
    train_matrix, validation_matrix = RatingMatrixDataset(["./ml-100k/u1.base"])

    # Run experiments
    for k in k_array:
        als_matrix_factorization(k, iterations, train_matrix, validation_matrix, True)


main()
