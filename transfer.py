import cv2
import os

import numpy as np
import plotly.graph_objs as go

from collections import namedtuple
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             recall_score,
                             precision_score)
from skimage.transform import resize
from tqdm import tqdm

# color hex codes for visuals
COLOR_SCHEME = {'cool_grey': "#8D99AE",
                'cherry_red': "#D90429",
                'navy': "#2B2D42",
                'white': "#EDF2F4",
                'bright_orange': '#FFA500',
                'mint': '#98FF98'}

NORM_PATH = '/Users/jackrisse/Datasets/Medical/chest-pneumonia/all/norm/'  # points to the imgs classified as normal
PNE_PATH = '/Users/jackrisse/Datasets/Medical/chest-pneumonia/all/Pneumonia/'  # points to the imgs classified as pneumonia
IMGS = '/Users/jackrisse/Capstone/imgs/'  # points to the new dir containing the augmented imgs
DATA = '/Users/jackrisse/Capstone/data/'
Metrics = namedtuple('Metrics', ['Accuracy', 'Recall', 'Precision'])

class_balancer = ImageDataGenerator(rotation_range=90,
                                    width_shift_range=100,
                                    height_shift_range=100,
                                    vertical_flip=True,
                                    horizontal_flip=True).flow_from_directory(NORM_PATH,
                                                                              target_size=(260,180),
                                                                              batch_size=317,
                                                                              color_mode='rgb',
                                                                              save_to_dir=IMGS,
                                                                              save_prefix='aug')
def balancer(imgDataGen):
    """
    class_balancer creates a generator that creates different normal chest xrays
    from the exsisting xrays to handle a class imbalance problem.
    """
    # calls the generator 5 times to double the num of normal imgs
    for i in range(5):
        next(imgDataGen)

def import_img():
    """
    The two loops are identical just pulls from different directories. The loop
    imports the file, reads the img, converts to a matrix and then appends the
    matrix to x and appends the classification to y.
    """
    x, y = [], [] # x contains the img arrays and y contains the labels corresponding to each img

    for f in tqdm(os.listdir(IMGS)):
        path = os.path.join(IMGS, f)
        img = cv2.imread(path)  # reads the file
        if img is not None:
            img = resize(img, (260,180,3))  # resizes the img
            img = np.asarray(img)  # converts to matrix
            x.append(img)  # adds to x
            y.append(0)  # adds to y

    for f in tqdm(os.listdir(PNE_PATH)):
        path = os.path.join(PNE_PATH, f)
        img = cv2.imread(path)
        if img is not None:
            img = resize(img, (260,180,3))
            img = np.asarray(img)
            x.append(img)
            y.append(1)

    x, y = np.asarray(x), np.asarray(y)  # converts x and y to numpy arrays
    return x, y

def train(xtrain, ytrain, optzr=None, val_set=None):
    # loads the MobileNet model
    base_model = MobileNet(include_top=False,
                            input_shape=(260,180,3),
                            weights='imagenet')

    x = base_model.output
    x = Flatten()(x)  # flattens the output array to be passed to a Dense layer
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)  # normalizes the output of the activation previous
    x = Dropout(0.5, seed=8)(x)  # sets a ratio of input units to zero to prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5, seed=8)(x)
    output = Dense(2, activation='softmax')(x)  # output layer

    # instantiating the model
    model = Model(inputs=base_model.input, outputs=output)

    # compiles the model with hyperparameters
    model.compile(optimizer=optzr,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    # stops the training process when validation loss stops decreasing
    early_stopper = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    # saves the model at the moment the model performs its best
    saver = ModelCheckpoint('./models/pneumonNet.h5',
                           monitor='val_loss',
                           save_best_only=True,
                           mode='min')
    # trains the model
    history = model.fit(x=xtrain,
                        y=ytrain,
                        batch_size=32,
                        epochs=5,
                        validation_data=val_set,
                        callbacks=[early_stopper, saver])

    return history, model


def crossval(x, y, folds):

    # loads the MobileNet model
    base_model = MobileNet(include_top=False,
                            input_shape=(260,180,3),
                            weights='imagenet')

    x = base_model.output
    x = Flatten()(x)  # flattens the output array to be passed to a Dense layer
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)  # normalizes the output of the activation previous
    x = Dropout(0.5, seed=8)(x)  # sets a ratio of input units to zero to prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5, seed=8)(x)
    output = Dense(2, activation='softmax')(x)  # output layer

    # instantiating the model
    model = Model(inputs=base_model.input, outputs=output)
    # compiles the model with hyperparameters
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    ytrue = []
    ypred = []
    # sets the kfolds obj
    kf = StratifiedKFold(folds, shuffle=True, random_state=8)

    fold = 0  # counter variable
    for train, val in kf.split(x):
        # simple print of what fold the training is at
        fold += 1
        print(f"Fold #{fold}")
        # spliting according to the Kfold obj
        xtrain, ytrain, xval, yval = x[train], y[train], x[val], y[val]
        ytrain, yval = to_categorical(ytrain), to_categorical(yval)
        # starts the train process for each fold
        model.fit(xtrain, ytrain, epochs=5, validation_data=(xval, yval))
        # predicts after each fold
        pred = model.predict(xval)
        # appends the predicted values
        ytrue.append(yval)
        ypred.append(pred)

    return np.array(ytrue), np.array(ypred)


def plot(history):
    acc, val_acc = history.history['binary_accuracy'], history.history['val_binary_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    x = history.epoch
    data = [go.Scatter(x=x, y=loss,
                        name='Training Loss',
                        line=dict(dash='solid',
                                  color=COLOR_SCHEME['mint'],
                                  width=3)),
            go.Scatter(x=x, y=val_loss,
                        name='Val Loss',
                        line=dict(dash='solid',
                                  color=COLOR_SCHEME['bright_orange'],
                                  width=3)),
            go.Scatter(x=x, y=acc,
                        name='Training Accuracy',
                        line=dict(dash='dash',
                                  color=COLOR_SCHEME['mint'],
                                  width=3)),
            go.Scatter(x=x, y=val_acc,
                        name='Val Accuracy',
                        line=dict(dash='dash',
                                  color=COLOR_SCHEME['bright_orange'],
                                  width=3))]
    layout = go.Layout(title='Training and Validation Metrics',
                        plot_bgcolor=COLOR_SCHEME['cool_grey'],
                        paper_bgcolor=COLOR_SCHEME['white'],
                        xaxis=dict(title='Epochs'),
                        yaxis=dict(title='Loss Metric'))
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def predict_w_thresh(ytest, ypred, threshold, _print=False):
    ytest = np.argmax(ytest, axis=1)

    ypreds = [1 if y[1] >= threshold else 0 for y in ypred]
    ypreds = np.asarray(ypreds)
    metrics = Metrics(accuracy_score(ytest, ypreds),
                      recall_score(ytest, ypreds),
                      precision_score(ytest, ypreds))
    if _print:
        print(metrics)

    return metrics

def test(filepath, model):
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            img = resize(img, (260, 180, 3))
            img = np.array([np.asarray(img)])

    prediction = model.predict(img)

    return prediction

def main():
    do_aug = input('Do augmentation? (y/n)\n').lower()
    import_or_load = input('Have images been converted? (y/n)\n').lower()
    what_optzr = input('Which optimizer? (adam/rms)\n').lower()
    do_cross_val = input('Cross Validate? (y/n)\n').lower()

    if do_aug == 'y':
        for f in os.listdir(IMGS):
            path = os.path.join(IMGS, f)
            # sanity check for me to see if any augmentation already happened
            if f.startswith('aug'):
                os.remove(f)  # removes any augmented file
        balancer(class_balancer)


    if import_or_load == 'n':
        x, y = import_img()
        # saves the data
        np.save('./data/x.npy', x)
        np.save('./data/y.npy', y)
        del x
        del y
        print('Data saved @ ./data')

    # loads the data
    x, y = np.load('./data/x.npy'), np.load('./data/y.npy')

    # use cross validation or just train straight up
    if do_cross_val == 'y':
        n_folds = input('How many folds?\n')  # asks how many folds
        int(n_folds)
        ytrue, ypred = crossval(x, y, n_folds)  # trains using cross validation
        for i in ytrue:
            # predicts with a threshold at 90%
            predict_w_thresh(ytrue, ypred, threshold=0.9, _print=True)

    else:
        # splits the data into train set, validation set, and test set
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=8)
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=.2, random_state=8)
        # converts the label features to compatable arrays for neural networks
        ytest = to_categorical(ytest, num_classes=2)
        ytrain = to_categorical(ytrain, num_classes=2)
        yval = to_categorical(yval, num_classes=2)
        # deletes x and y because they are not needed anymore
        del x
        del y
        # decides which optimizers to use
        if what_optzr=='adam':
            optzr = 'adam'
        elif what_optzr=='rms':
            optzr = RMSprop(lr=0.0001)
        # trains the model
        history, model = train(xtrain, ytrain,
                               optzr=optzr,
                               val_set=(xval, yval))

        plot(history)



if __name__ == '__main__':
    main()
