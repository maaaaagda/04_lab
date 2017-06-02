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

def extract(x):
    return np.array(np.sum(x, axis=1))

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
cut_off = np.round(np.shape(X_train)[0]*0.1)        #30000
cut_off_2 = np.round(np.shape(X_train)[1] * 0.05)         #3000
X_train = X_train[15000-cut_off: 15000+cut_off, 1500-cut_off_2:1500+cut_off_2]
Y_train = Y_train[15000-cut_off: 15000+cut_off, :]




g = np.reshape(Y_train, np.shape(Y_train)[0], 1)

X = X_train
y = one_hot(g)

w = np.zeros((cut_off*2, 56))
tt =cut_off.astype(np.int64)*2
for u in range(tt):
    w[u, :]=(extract(np. reshape(data[0][u, :], (56, 56))))
#print(w)

ww = np.array([extract(X_train)]).T
pol = len(ww)
#print(pol)
#print(extract(X_train))
np.random.seed(1)

syn0 = 2*np.random.random((56, 15)) -1
syn1 = np.random.random((15, 36)) - 1
for i in range(300):
        l0 = w
        l1 = sigma(np.dot(l0, syn0))
        l2 = sigma(np.dot(l1, syn1))
        l2_error = y - l2
        l2_delta = l2_error * sigma(l2, True)/1000

        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigma(l1, True)/1000
        syn1 += l1.T.dot(l2_delta)
        syn0 += 10*np.dot(l0.T, l1_delta)
print("hej")
result = np.round(l2)
b  = np.argmax(l2, axis=1)
#print(result)

def prediction(predicted, real):
    return np.sum(predicted*real)/np.shape((predicted)[0])



print(b)
#print(g)
print(np.sum(b==g)/len(b))
print(time.time()-start_time)
"""
a = np.array([1, 0, 36])
c = a.T
g = np.reshape(Y_train, np.shape(Y_train)[0], 1)
print (g)
print(one_hot(g))
"""