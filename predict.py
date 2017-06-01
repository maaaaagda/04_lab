# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
#import cv2
#import time
#start_time = time.time()


def load_data():
    FILE_PATH = 'train.pkl'
    with open(FILE_PATH, 'rb') as f:
        return pkl.load(f)

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    #warnings.filterwarnings('ignore')
    data = load_data()
    X_train = data[0]
    Y_train = data[1]
    length = len(X_train)
    cut_off = length * 0.18
    X_train = X_train[0: cut_off, 0:cut_off]
    Y_train = Y_train[0: cut_off, :]
    #dist = hamming_distance(x[:, 0:cut_off], X_train)
    #return sort_train_labels_knn(dist, Y_train)
    pass


def sigma(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
"""
data = load_data()
X_train = data[0]
Y_train = data[1]
length = len(X_train)
cut_off = length * 0.18
X_train = X_train[0: cut_off, 0:cut_off]
Y_train = Y_train[0: cut_off, :]
"""
X = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])
y = np.array([[1, 0, 0], [0,1,0], [0,0,1], [0,0,1]])


np.random.seed(1)

syn0 = 2 * np.random.random((4, 3)) - 1
for i in range(10000):
    l0 = X
    l1 = sigma(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * sigma(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

print(syn0)
print("hej")
print(np.round(l1))
print(np.round(sigma(np.dot(np.array([1,1,1,1]),syn0))))