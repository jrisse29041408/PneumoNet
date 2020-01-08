import os
import numpy as np

from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
# from transfer import predict_w_thresh

# loads the data
# x, y = np.load('./data/x.npy'), np.load('./data/y.npy')
# # splits the data into train set, validation set, and test set
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=8)
# xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=.2, random_state=8)
# # converts the label features to compatable arrays for neural networks
# ytest = to_categorical(ytest, num_classes=2)
# ytrain = to_categorical(ytrain, num_classes=2)
# yval = to_categorical(yval, num_classes=2)
# # deletes x and y because they are not needed anymore
# del x
# del y

model = load_model('models/pneumonNet.h5')  # loading the saved model
plot_model(model, to_file='./models/visual_model')
# ypred = model.predict(xtest)  # predicting
#
# predict_w_thresh(ytest, ypred, 0.9)  # predicting with the threshold set at 90%
