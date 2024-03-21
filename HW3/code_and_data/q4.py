'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels): 
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(10):
        X = data.get_digits_by_label(train_data, train_labels, k)
        means[k] = np.sum(X, axis=0) / X.shape[0]
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        X = data.get_digits_by_label(train_data, train_labels, k)
        covariances[k] = ((X - means[k]).T @ (X - means[k])) / X.shape[0] + 0.01 * np.identity(64)
    return covariances

def generative_likelihood(digits, means, covariances): 
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    exp = np.zeros(shape=(len(digits), means.shape[0]))
    const = -means.shape[1]/2 * np.log(2 * np.pi) - 1/2 * np.log(np.linalg.det(covariances))

    for j in range(means.shape[0]):
        exp[:, j] += -1/2 * np.diag((digits - means[j]) @ np.linalg.inv(covariances[j]) @ (digits - means[j]).T)
    re_val = const + exp
    return re_val
    

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    al = generative_likelihood(digits, means, covariances) + np.log(0.1)
    pk = logsumexp(al, axis=1).reshape(-1, 1)
    re_val = al - pk
    return re_val

def avg_conditional_likelihood(digits, labels, means, covariances): 
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    total = 0
    c = digits.shape[0]
    for j in range(digits.shape[0]):
        label = int(labels[j])
        total += cond_likelihood[j, label]
    re_val = total/c
    return re_val

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

    train_avg_log = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_log = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    train_acc = np.mean(classify_data(train_data, means, covariances) == train_labels)
    test_acc = np.mean(classify_data(test_data, means, covariances) == test_labels)

    print(f"Train average conditional log-likelihood: {train_avg_log}.")
    print(f"Test average conditional log-likelihood: {test_avg_log}.")
    print(f"Train accuracy:  {train_acc}.")
    print(f"Test accuracy: {test_acc}.")

    # plot the leading eigenvectors for each class covariance matrix side by side as 8x8 images
    v = []
    for i in range(10):
        eigenvalues, eigenvectors = np.linalg.eig(covariances[i])
        v.append(eigenvectors[:, np.argmax(eigenvalues)].reshape(8, 8))
    img = np.concatenate(v, axis=1)
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
