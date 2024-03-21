from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user = data['user_id']
    question = data['question_id']
    correct = data['is_correct']
    diff = np.array([theta[i] for i in user]) - np.array([beta[j] for j in question])
    log_lklihood = np.log(sigmoid(diff)) * correct + np.log(1 - sigmoid(diff)) * (1 - np.array(correct))
    log_lklihood = sum(log_lklihood)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user = np.array(data['user_id'])
    question = np.array(data['question_id'])
    correct = np.array(data['is_correct'])

    # update theta
    partial_theta = np.zeros(len(theta))
    for i in range(len(correct)):
        diff = theta[user[i]] - beta[question[i]]
        partial_theta[user[i]] -= correct[i] - sigmoid(diff)
    theta = theta - lr * partial_theta

    # update beta
    partial_beta = np.zeros(len(beta))
    for i in range(len(correct)):
        diff = theta[user[i]] - beta[question[i]]
        partial_beta[question[i]] -= sigmoid(diff) - correct[i]
    beta = beta - lr * partial_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    train_neg_lld = []
    val_neg_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_neg_lld.append(neg_lld)
        val_neg_lld.append(neg_lld_val)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_neg_lld, val_neg_lld


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    num_iteration = 50
    theta, beta, val_acc_lst, train_neg_lld, val_neg_lld = irt(train_data, val_data, lr, num_iteration)

    iterations = [i for i in range(num_iteration)]
    plt.plot(iterations, train_neg_lld, label='training neg log-likelihoods')
    plt.plot(iterations, val_neg_lld, label='validation neg log-likelihoods')
    plt.title('lr={}, num_iteration={}'.format(lr, num_iteration))
    plt.xlabel('number of iterations')
    plt.ylabel('negative log-likelihoods')
    plt.show()

    # 2c
    val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    print('Final validation accuracy: {}'.format(val_acc))
    print('Final test accuracy: {}'.format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    q1, q2, q3 = 100, 500, 1000
    p1 = sigmoid(theta - beta[q1])
    p2 = sigmoid(theta - beta[q2])
    p3 = sigmoid(theta - beta[q3])
    plt.scatter(theta, p1, label='{}-th question'.format(q1))
    plt.scatter(theta, p2, label='{}-th question'.format(q2))
    plt.scatter(theta, p3, label='{}-th question'.format(q3))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
