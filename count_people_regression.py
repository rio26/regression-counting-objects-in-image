import numpy as np
import matplotlib.pyplot as plt
from os import path




# file path
file_trainx, file_trainy = path.relpath("data/count_data_trainx.txt"), path.relpath("data/count_data_trainy.txt")
file_testx, file_testy = path.relpath("data/count_data_testx.txt"), path.relpath("data/count_data_testy.txt")

# read data
data_trainx, data_trainy = np.genfromtxt(file_trainx, delimiter="\t").transpose(), np.genfromtxt(file_trainy, delimiter="\t").transpose()
data_testx, data_testy = np.genfromtxt(file_testx, delimiter="\t").transpose(), np.genfromtxt(file_testy, delimiter="\t").transpose()

print(data_trainx.shape, data_trainy.shape)
print("Done.")
