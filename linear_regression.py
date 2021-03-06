import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
#import sklearn.cross_validation(might be obsolete)
from sklearn.model_selection import train_test_split


###by: Aashish Dugar

#######################################
#### Normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    x1 = np.array(train)  # for ease of operations, treat it as an numpy array
    x2 = np.array(test)  # same here
    train_normalized = (x1 - x1.min(0)) / x1.ptp(0)
    """ptp represents a point to point function which essentially takes the max and
    min and subtracts them"""
    test_normalized = (x2 - x1.min(0)) / x1.ptp(0)
    # using the same function/operations from train set.
    return train_normalized, test_normalized  # return the values of the set.


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    loss = 0  # initialize the square_loss
    s1 = np.matmul(np.transpose(theta), X)  # multiply the parameter vector
    s2 = s1 - y  # subtract from true value to get cost function
    s3 = np.square(s2)  # square each term to get square loss
    size = y.size  # no. of instances or samples
    s4 = np.sum(s3)  # add all the elements of the array, basically the summation step.
    loss = s4 / size  # "1/m", i.e dividing the loss by the size
    return loss  # final value of J





########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    X_transpose = np.transpose(X)
    s1 = np.matmult(X,theta)
    s2 = s1 - y
    s3 = np.matmult(X_transpose,s2)
    s4 = (2 * s3)/y.size
    grad = s4

    return grad




####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    batch gradient descent to minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeroes(num_features) #initialize theta
    X_transpose = X.transpose()
    for iter in range(0,num_iter):
        hypothesis = np.dot(X,theta)
        loss = hypothesis - y
        J = np.sum(loss**2)/num_instances
        gradient = np.dot(X_transpose,loss)/num_instances
        theta = theta - alpha * gradient
        loss_hist[iter + 1] = J
        theta_hist = (np.arange(num_iter + 1), theta)
    return theta_hist, loss_hist



###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    s1 = np.dot(X,theta)
    s2 = s1 - y
    J = ((np.sum(s2**2))/y.size) + ((lambda_reg) * (np.transpose(theta),theta))
    return J;

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    m = y.size
    X_transpose = np.transpose(X)
    for iter in range(0,num_iter):
        s1 = np.dot(X,theta)
        residual = s1 - y
        J = ((np.sum(residual ** 2)) / num_instances) + (lambda_reg) * np.dot(np.transpose(theta),theta)
        gradient = (np.dot(X_transpose, residual) / m)
        theta = theta - alpha * (gradient + 2 * (lambda_reg / m) * theta)
        loss_hist[iter + 1] = J
    theta_hist = (np.arange(num_iter+1),theta)
    return theta_hist,loss_hist



#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta


    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    



def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term

    # add code to print necessary values here.

if __name__ == "__main__":
    main()
