'''
Created on Oct 10th, 2018
@author: Rio Li
'''

from cvxopt import matrix
from cvxopt import solvers
import numpy as np
from numpy.dual import inv
import matplotlib.pyplot as plt

from os import path
import logging

from regressions import*

# generate a matrix with each column is a polynomial of x
def generateOrderMatrix(vector, order):
    # print("order: ", order,"\n")
    col = np.copy(vector) # original sample x
    matrix = np.ones((vector.size , 1)) # generate the target matrix
    # print("before loop: ", matrix.shape , "\n")


    for i in range(1 , order +1 ):
        icol = pow(col, i)
        matrix = np.c_[matrix, icol]
        # print("in the loop: ", matrix.shape)

    # print("end of the loop: ", matrix.shape)
    return matrix.T

# least square traning
def train_LS(xs, ys, xp, yp, order):
    x_train = np.array(generateOrderMatrix(xs, order))  # (6,50)
    x_test = np.array(generateOrderMatrix(xp, order))   # (6,100)
    y_train = ys # (1,50)
    y_test = yp # (1,100)


    #---------------------------------Least Squre---------------------------------#
    theta_ls =  inv(x_train.dot(x_train.T)).dot(x_train).dot(y_train) # use sample x and sample y to obtain theta
    y_ls = x_test.T.dot(theta_ls)  # use the obtained theta on test x to obtain predictor new_y 
    mse = mean_square_error(y_test, y_ls)  # calculate the error between actual y and new_y
    plt.plot(xp, y_ls, color = 'b', label='LS')
    

    #---------------------------------Regularised Least Squre---------------------------------#
    theta_rls =  inv(x_train.dot(x_train.T)+ 1).dot(x_train).dot(y_train)
    y_rls = x_test.T.dot(theta_rls) 
    mse_rls = mean_square_error(y_test, y_rls) 
    plt.plot(xp, y_rls, color = 'g', label='RLS')
    
    #---------------------------------LassoRegression---------------------------------#

    #---------------------------------RobustRegression---------------------------------#

    #---------------------------------BayesianRegression---------------------------------#
    alpha = 0.48
    sigma = 1
    # rxr
    theta_sigma = inv((1/alpha) * np.identity(x_train.shape[0]) + (1/sigma) * np.dot(x_train, x_train.T))
    theta_mu = (1/sigma)* np.dot(theta_sigma, np.dot(x_train, y_train)) # rxr
    y_beyesian = x_test.T.dot(theta_mu)
    print(y_beyesian.shape)
    mse_rls = mean_square_error(y_test, y_beyesian)
    plt.plot(xp, y_beyesian, color = 'k', label='Bayesian')


    #---------------------------------Plot---------------------------------#
    plt.legend(['LS', 'RLS', "Beyesian"], loc='upper right')
    plt.show()

# This function plots the true polynomial function and the sample points.
# x,y are the sample points
# tx,ty are the input values for the true function
def plot_true_function(x, y, tx, ty, theta):
    # plot the sample points (sampx, samply) as scatter plot
    plt.scatter(x, y, marker='.')

    # predicted response vector
    y_pred = theta[0] + theta[1] * x + theta[2] * (x ** 2) + theta[3] * (x ** 3) + theta[4] * (x ** 4) + theta[5] * (x ** 5)

    # plotting the true polynomial function
    plt.plot(tx, ty, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(('samples', 'function'), loc='upper right')

    # function to show plot
    plt.show()


def mean_square_error(actual, predictor):
    return np.square(actual - predictor).mean()

# -----------------------------------------TEST----------------------------------------- #
# Uncomment the following line when debugging:
# logging.basicConfig(level=logging.INFO)


# file path
file_sampx, file_sampy = path.relpath("data/polydata_data_sampx.txt"),path.relpath("data/polydata_data_sampy.txt")
file_polyx, file_polyy = path.relpath("data/polydata_data_polyx.txt"), path.relpath("data/polydata_data_polyy.txt")
file_thtrue = path.relpath("data/polydata_data_thtrue.txt")

# read data: sample x,y; poly x,y; true theta as tt
with open(file_sampx, 'r') as sx, open(file_sampy, 'r') as sy, open(file_polyx, 'r') as px, open(file_polyy, 'r') as py , open(file_sampy, 'r') as t:
    sx, sy, px, py, tt = np.genfromtxt(sx), np.genfromtxt(sy), np.genfromtxt(px), np.genfromtxt(py), np.genfromtxt(t)

# sy,py = sy.reshape((-1, 1)).T, py.reshape((-1, 1)).T
# print(sx.shape, sy.shape, px.shape, py.shape, tt.shape)

# x_train = np.array(generateOrderMatrix(sx, 5))  
# x_test = np.array(generateOrderMatrix(px, 5)) 
# print(x_train.size, x_test.size)
# print(x_train)
train_LS(sx, sy, px, py, 5)

print("Done.")  
