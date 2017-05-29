"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from sklearn import linear_model
# Import datasets, classifiers and performance metrics   datasets,
from sklearn import svm, metrics
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import time
start_time = time.time()
def load_data():
    FILE_PATH = 'train.pkl'
    with open(FILE_PATH, 'rb') as f:
        return pkl.load(f)

# The digits dataset
#digits = datasets.load_digits()
datas = load_data()
X_train = datas[0]
Y_train = datas[1]
length = len(X_train)
cut_off = round(length )
fr = 1000
to = 1900

images = X_train[0: cut_off, :]
#for i in images:
 #   i = np.reshape(i, (56,56) )
target = Y_train[0: cut_off, :]
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(images, target))

for index, (image, label) in enumerate(images_and_labels[17:22]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    image =  np.reshape(image, (56, 56))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)
data = images #[:, 1000:1900] #images.reshape((n_samples, -1))
cut_off = 1.5
# Create a classifier: a support vector classifier
#classifier = svm.SVC(gamma=0.001)
classifier = RandomForestClassifier(min_samples_leaf=20)
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / cut_off],target[:n_samples / cut_off])
#classifier = svm.LinearSVC() #_model.SGDRegressor()
#classifier.fit( X = data[:n_samples / 2], y=target[:n_samples / 2])
# Now predict the value of the digit on the second half:
expected = target[n_samples /cut_off:]
predicted = classifier.predict(data[n_samples / cut_off:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted))
      )
print("Confusion matrix:\n%s"
      #% metrics.confusion_matrix(expected, predicted)
      )

images_and_predictions = list(zip(images[n_samples / cut_off:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    image = np.reshape(image, (56, 56))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()



