# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum, x_train, y_train, tau, lam = 1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    test_x_row = np.array([test_datum])

    norms = l2(test_x_row, x_train) #1 * 354
    pow = - norms / (2 * (tau ** 2))
    A = np.exp(pow - logsumexp(pow))

    X_T = x_train.transpose()
    X, y = x_train, y_train


    l = ((X_T * A) @ X) + lam
    r = (X_T * A) @ y


    try:
        W = np.linalg.solve(l, r)
    except:
        W = np.linalg.pinv(l) @ r

    y_hat = np.dot(W, test_datum)

    return y_hat

    ## TODO



def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    ids = np.arange(x.shape[0])
    np.random.shuffle(ids)

    x_sh, y_sh = x[ids], y[ids]
    split = int(x.shape[0] * (1 - val_frac))

    x_train = x_sh[:split]
    y_train = y_sh[:split]
    x_valid = x_sh[split:]
    y_valid = y_sh[split:]

    valid_losses = np.empty_like(taus)

    for (i, t) in enumerate(taus):

        valid_pre = np.array([
            LRLS(val, x_train, y_train, t)
            for val in x_valid
        ])

        valid_errs_sq = ((valid_pre - y_valid) ** 2)
        valid_losses[i] = np.mean(valid_errs_sq)

    return valid_losses

    ## TODO


if __name__ == "__main__":

    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    # print(test_losses)
    # plt.semilogx(train_losses)
    split = 30
    plt.semilogx(taus[:split], test_losses[:split])
    plt.xlabel('tau)')
    plt.ylabel("loss")
    plt.show()

    plt.semilogx(taus[split:], test_losses[split:])
    plt.xlabel('tau)')
    plt.ylabel("loss")
    plt.show()


    # # taus = np.arange(10, 1000)
    # taus = np.logspace(1.0,3,200)
    # valid_losses = run_validation(x,y,taus,val_frac=0.3)
    # log_taus = np.log(taus)
    #
    # plt.plot(taus,valid_losses)
    # # plt.plot(valid_ce, itera, label = "valid")
    # plt.xlabel('tau)')
    # plt.ylabel("loss")
    # # plt.title('mnist_train')
    # plt.legend()
    # plt.show()


