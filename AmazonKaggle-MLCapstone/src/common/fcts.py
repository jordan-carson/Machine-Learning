import numpy as np
import time
import pandas as pd
import os
import socket
import logging
import logging.handlers
import sys
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt

# Deep Learning imports
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score
from keras.models import Model
from sklearn.cross_validation import KFold


# init_logger('~/PycharmProjects/AmazonKaggle-MLCapstone/Logs', 'Utility')

def plot_pictures(label, df_train, train_path):

    images = df_train[df_train[label] == 1].image_name.values

    fig , ax = plt.subplots(nrows=3, ncols=3, figsize=(8,8))
    ax = ax.flatten()

    for i in range(0,9):
        f = random.choice(images)
        img = Image.open(os.path.join(train_path, f + '.jpg'))
        ax[i].imshow(img)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title("{}s h:{}s w:{}s".format(f, img.height, img.width))
    plt.tight_layout()


def init_logger(log_dir, process_name, loglevel_file=20, loglevel_stdout=40):
    logging.getLogger().setLevel(logging.NOTSET)  # 0
    logging.getLogger().handlers = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    logfile_name = process_name + '_' + str(socket.gethostname()) + '_' + str(os.getenv('Username')) \
                   + '_' + timestamp + '.txt'
    logfile_path = os.path.join(log_dir, logfile_name)

    # create directory if it doesn't already exist
    try:
        os.makedirs(log_dir)
    except:
        pass

    # delete previous version of this logfile if it exists

    try:
        os.remove(logfile_name)
    except:
        pass

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # file handler
    file_h = logging.handlers.RotatingFileHandler(logfile_path, mode='w', maxBytes=50*1024*1024,
                                                  backupCount=2)
    file_h.setLevel(loglevel_file)  # 20 = info
    file_h.setFormatter(formatter)

    # stdout handler
    stdout_h = logging.StreamHandler(stream=sys.stdout)
    stdout_h.setLevel(loglevel_stdout)  # 40 = error
    stdout_h.setFormatter(formatter)
    logging.getLogger().addHandler(stdout_h)

    logging.info('Logging initialised. PID ' + str(os.getpid()))


def get_optimal_threshhold(true_label, prediction, iterations = 100):

    best_threshhold = [0.2]*17
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value


def create_model_vgg16(image_dimensions=(128, 128, 3)):
    input_tensor = Input(shape=image_dimensions)
    bn = BatchNormalization()(input_tensor)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions)
    x = base_model()(bn)
    x = Flatten()(x)
    output = Dense(17, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model


def fbeta(true_label, prediction):
   return fbeta_score(true_label, prediction, beta=2, average='samples')


def fbeta_2(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')


def n_crossvalidation(nfolds, num_fold, sum_score, y_train, x_train, y_test, x_test):
    logging.info("creating cross validation before building the network model")
    yfull_test = []
    yfull_train = []

    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=18)

    for train_index, test_index in kf:
        start_time_model_fitting = time.time()

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

        model = Sequential()
        model.add(BatchNormalization(input_shape=(32, 32, 3)))
        model.add(Conv2D(8, 1, 1, activation='relu'))
        model.add(Conv2D(16, 2, 2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=128, verbose=2, nb_epoch=10, callbacks=callbacks,
                  shuffle=True)

        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        p_valid = model.predict(X_valid, batch_size=128, verbose=2)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

        p_test = model.predict(x_train, batch_size=128, verbose=2)
        yfull_train.append(p_test)

        p_test = model.predict(x_test, batch_size=128, verbose=2)
        yfull_test.append(p_test)





