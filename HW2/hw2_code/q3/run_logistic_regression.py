from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.1,########?????
        "num_iterations": 600
    }
    weights = np.full(shape = (M + 1, 1), fill_value = 0, dtype = np.float64)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    print('\n \n \n')
    # init_cost = 0
    # previous_cost = 1
    itera = []
    train_ce = []
    valid_ce = []

    for t in range(hyperparameters["num_iterations"]):
        cost, ders, pros =  logistic(weights, train_inputs, train_targets, hyperparameters)
        # if t == 0:
        #     init_cost = cost
        # if t % 5 == 0:
        #     improved_cost = previous_cost - cost
        #     previous_cost = cost
        #     print("iter = {:>2}, cost = {:>4}, improved_cost = ,{}".format(t, cost, improved_cost))
        #     if improved_cost < 0:
        #         print("\n \n geting worse \n \n")
        weights -= (hyperparameters["learning_rate"]) * ders


        itera.append(t)

        train_y = logistic_predict(weights, train_inputs)
        train_ce.append(evaluate(train_targets, train_y)[0])

        valid_y = logistic_predict(weights, valid_inputs)
        valid_ce.append(evaluate(valid_targets, valid_y)[0])


        # print("\n total cost improvement = ", init_cost - previous_cost, '\n\n\n')

    # print(hyperparameters)

#-------------------------below for 3.2c----------------------------------
    # plt.plot(itera, train_ce, label = "train")
    # plt.plot(itera, valid_ce, label = "valid")
    # plt.xlabel('iteration')
    # plt.ylabel("cross entropy")
    # plt.title('mnist_train_small')
    # plt.legend()
    # plt.show()

#----------------------------below for 3.2b--------------------------------
    # train_y = logistic_predict(weights, train_inputs)
    # train_fce, train_class_error = evaluate(train_targets, train_y)
    #
    # valid_y = logistic_predict(weights, valid_inputs)
    # valid_fce, valid_class_error =evaluate(valid_targets, valid_y)
    #
    # test_y = logistic_predict(weights, test_inputs)
    # test_fce, test_class_error =evaluate(test_targets, test_y)
    #
    # print("learning rate: ", hyperparameters["learning_rate"], "num_iterations: ", hyperparameters["num_iterations"], '\n')
    # print("train_fce: ",train_fce, "train_class_error: " ,1-train_class_error , '\n')
    # print("valid_fce: ",valid_fce, "valid_class_error: " ,1-valid_class_error , '\n')
    # print("test_fce: ",test_fce, "test_class_error: " ,1-test_class_error , '\n')
    #
    # print("-----------------------------------------------------------------------")










    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()

