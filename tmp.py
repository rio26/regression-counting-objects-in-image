import numpy as np
from numpy.dual import inv
import matplotlib.pyplot as plt


# c0 = [7,8]
# c1 = [0,0]
# print("c0: ", c0)
# print("c1: ", c1)
# app2 = np.append(c0,c1)
# print("append2 = c0+c1: ", app2, "type:", type(app2), "size: ", app2.size, "\n")


m1 = np.zeros((3, 1))
print("m1", m1, "size: ", m1.size, "type:", type(m1), m1.shape)
m2 = np.ones((3 , 1))
print("m2", m2, "size: ", m2.size, "type:",type(m2), m2.shape)
m1 = np.c_[m1, m2]
print("----------------------after 1st cat----------------------------")
print("m1", m1, "size: ", m1.size, "type:", type(m1))
print("m2", m2, "size: ", m2.size, "type:",type(m2))
print("----------------------after 2nd cat----------------------------")
m1 = np.c_[m1, m2]
print("m1", m1, "size: ", m1.size, "type:", type(m1))