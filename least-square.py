import numpy as np
from numpy.dual import inv
import matplotlib.pyplot as plt
from os import path

# parameter estimate theta for least square
def theta_least_square(x, y):
    return inv(x.dot(y.T)).dot(x).dot(y)

# parameter estimate theta for regularized least square
def theta_regularized_ls(x, y):
    return inv(x.dot(y.T)).dot(x).dot(y)

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



# file path
file_sampx, file_sampy = path.relpath("data/polydata_data_sampx.txt"),path.relpath("data/polydata_data_sampy.txt")
file_polyx, file_polyy = path.relpath("data/polydata_data_polyx.txt"), path.relpath("data/polydata_data_polyy.txt")
file_thtrue = path.relpath("data/polydata_data_thtrue.txt")

# read data: sample x,y; poly x,y; true theta as tt
with open(file_sampx, 'r') as sx, open(file_sampy, 'r') as sy, open(file_polyx, 'r') as px, open(file_polyy, 'r') as py , open(file_sampy, 'r') as t:
    sx, sy, px, py, tt = np.genfromtxt(sx), np.genfromtxt(sy), np.genfromtxt(px), np.genfromtxt(py), np.genfromtxt(t)


plot_true_function(sx, sy, px, py, tt)

# print(sx, sy)
print("Done.")



# # observations
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
#
# # estimating coefficients
# b = estimate_coef(x, y)
# print("Estimated coefficients:\nb_0 = {}  \
# \nb_1 = {}".format(b[0], b[1]))
#
# # plotting regression line
# plot_regression_line(x, y, b)
#
