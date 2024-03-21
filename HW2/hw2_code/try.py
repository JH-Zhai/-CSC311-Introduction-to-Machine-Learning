import numpy as np
from q3.logistic import *
from q4 import *
from q3.utils import sigmoid
# from q3.utils import *

def load_train():
    """ Loads training data.
    """
    with open('q3/data/mnist_train.npz', 'rb') as f:
        train_set = np.load(f)
        train_inputs = train_set['train_inputs']
        train_targets = train_set['train_targets']

    return train_inputs, train_targets


def load_train_small():
    """ Loads small training data.
    """
    with open('q3/data/mnist_train_small.npz', 'rb') as f:
        train_set_small = np.load(f)
        train_inputs_small = train_set_small['train_inputs_small']
        train_targets_small = train_set_small['train_targets_small']
        # print(train_targets_small)
        # print(train_targets_small.size)
        # for i in range(train_targets_small.size):
        #     print(train_targets_small[i])
        # print((train_targets_small == train_targets_small)[2] == False)



    return train_inputs_small, train_targets_small


def load_valid():
    """ Loads validation data.
    """
    with open('q3/data/mnist_valid.npz', 'rb') as f:
        valid_set = np.load(f)
        valid_inputs = valid_set['valid_inputs']
        valid_targets = valid_set['valid_targets']

    return valid_inputs, valid_targets


def load_test():
    """ Loads test data.
    """
    with open('q3/data/mnist_test.npz', 'rb') as f:
        test_set = np.load(f)
        test_inputs = test_set['test_inputs']
        test_targets = test_set['test_targets']

    return test_inputs, test_targets


if __name__ == "__main__":

    train_inputs, train_targets = load_train()
    train_small_inputs, train_small_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    a = np.array([1,2,3,4])
    b = np.array([1,1,1,1])
    d = np.diag(a)
    print(np.dot(a,b))

    # num_examples = 20
    # num_dimensions = 10
    #
    # weights = np.random.randn(num_dimensions + 1, 1)
    # data = np.random.randn(num_examples, num_dimensions)
    # targets = np.random.rand(num_examples, 1)
    #
    # f, df, y = logistic(weights, data, targets, None)
    # print(f, '\n', '\n')
    # print(df, '\n', '\n')
    # print(y, '\n', '\n')
    # extra_column = np.full(shape = 4, fill_value = 1, dtype = int)
    # print(extra_column)






    # N = 10
    # a = np.random.rand(N,N)
    # data = np.array([
    #     [2, 2, 4, 3],
    #     [2, 4, 4, 2],
    #     [4, 5, 6, 5]
    # ])
    # extra_column = np.full(shape = (3, 1), fill_value = 1.23, dtype = np.float64)
    # extra_column[2][0] += 0.023
    # data_ext = np.column_stack((data, extra_column))
    # print(data_ext)
    # # N, M = data.shape
    # # print(N, M)
    # # print()
    # weight = np.array([[-5,5,-5,5,2]]).transpose()
    # print(weight, '\n')
    # weight += 2 * np.array([[-1,2,-6,2,9]]).transpose()
    #
    # print(weight)
    # names = ['David', 'Peter', 'Michael', 'John', 'Bob']
    # for i in range (len (names)):
    #     print("{}.{}".format(names[i], i + 1))

    # y = logistic_predict(weight, data)
    # print(y)
    # targets = np.array([[0], [1], [1]])

    # f, df, y = logistic(weight, data, targets, None)
    # print(f)
    # print()
    # print(df)
    # print()
    # print(y)
    # a = float(y[1][0])
    # print(a)
    # ce, frac_correct = evaluate(targets, y)
    # print(ce, frac_correct)



    # # k = (c > 0.5).astype(int)
    # k = np.around(y).astype(int)
    # p = np.around(y).astype(int)
    # # p[0][0] = 1
    # print(k)
    # print(p)
    # for i in range(k.size):
    #     if k[i][0] != p[i][0]:
    #         print("kkk")
    #
    # mapped = sigmoid(c)
    # k = int(c[0][0])

    # print(a)
    # print(b)
    # print()
    # print(c)
    # print(mapped)
    # colum_to_add = np.full(shape=3, fill_value=1, dtype=int)
    # print(colum_to_add)
    # a = np.column_stack((a, colum_to_add))
    # print(a)
    #
    # row , column = a.shape
    # print(row)
    # print(column)
    #
    # num_dimensions = 5
    # num_examples = 10
    # weights = np.random.randn(num_dimensions + 1, 1)
    # data = np.random.randn(num_examples, num_dimensions)
    # print(type(weights))
    # print(type(data))
    # b = np.zeros((N,N+1))
    # print(weights)
    # print(data)

