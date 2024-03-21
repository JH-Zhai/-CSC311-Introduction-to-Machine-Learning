from q3.utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    y = None
    N, M = data.shape
    extra_column = np.full(shape = N, fill_value = 1, dtype = int)
    data_ext = np.column_stack((data, extra_column))
    z = np.dot(data_ext, weights)
    y = sigmoid(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    ce = None
    frac_correct = None
    N = targets.size
    y_labels = np.around(y).astype(int)
    incorrect_count = 0
    for i in range(N):
        if y_labels[i][0] != targets[i][0]:
            incorrect_count += 1
    frac_correct = float(1.0 - float(incorrect_count) / (N * 1.0))
    total_ce = float(0)
    for i in range(N):
        # print(type(y[i][0]))
        y_single = y[i][0]
        t = np.float64(targets[i][0])
        single_ce = float((-t) * np.log(y_single) - (1-t) * np.log(1-y_single))
        total_ce += single_ce
    ce = float(total_ce / (N * 1.0))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f = None
    df = None
    N, M = data.shape
    extra_column = np.full(shape = N, fill_value = 1, dtype = np.float64)
    data_ext = np.column_stack((data, extra_column))
    f, frac_correct = evaluate(targets, y)
    ders = []
    for j in range(M + 1):
        total_der_j = np.float64(0)
        for i in range(N):
            der = (y[i][0] - targets[i][0]) * data_ext[i][j]
            total_der_j += der
        ders.append([total_der_j / N])
    df = np.array(ders)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
