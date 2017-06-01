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
import time
start_time = time.time()
np.set_printoptions(threshold=np.nan)

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

def one_hot(a):
    b = np.zeros((len(a), 36))
    b[np.arange(len(a)), a] = 1
    return b

data = load_data()
X_train = data[0]
Y_train = data[1]
length = len(X_train)
cut_off = np.round(np.shape(X_train)[0]*0.05)        #30000
cut_off_2 = np.round(np.shape(X_train)[1] * 0.05)         #3000
X_train = X_train[15000-cut_off: 15000+cut_off, 1500-cut_off_2:1500+cut_off_2]
Y_train = Y_train[15000-cut_off: 15000+cut_off, :]



g = np.reshape(Y_train, np.shape(Y_train)[0], 1)

X = X_train
y = one_hot(g)

np.random.seed(1)

syn0 = 2 * np.random.random((cut_off_2*2, 36)) - 1

for i in range(1000):
    l0 = X
    l1 = sigma(np.dot(l0, syn0))
    """print("syn0")
    print(syn0)
    print("l1")
    print(l1)"""
    l1_error = y - l1
    l1_delta = l1_error * sigma(l1, True)
    syn0 += np.dot(l0.T, l1_delta)



print("hej")
result = np.round(l1)
b  = np.argmax(l1, axis=1)
#print(result)

def prediction(predicted, real):
    return np.sum(predicted*real)/np.shape((predicted)[0])


print(np.sum(b==g))
print(b)
print(g)
print(time.time()-start_time)
"""
a = np.array([1, 0, 36])
c = a.T
g = np.reshape(Y_train, np.shape(Y_train)[0], 1)
print (g)
print(one_hot(g))
"""
