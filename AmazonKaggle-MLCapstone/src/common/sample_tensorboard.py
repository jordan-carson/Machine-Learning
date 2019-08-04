import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
import cv2
import tensorflow as tf
from tqdm import tqdm
import time
from src.common.fcts import init_logger
from src.common.fcts import optimise_f2_thresholds
from src.common.fcts import get_optimal_threshhold
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import pydot, graphviz
from pydot import print_function
import random
from PIL import Image
import matplotlib.pyplot as plt


init_logger('/Users/jordancarson/Logs', 'image_processing')

MAIN = '/Users/jordancarson/PycharmProjects/AmazonKaggle-MLCapstone/resources'
TRAIN_LABELS = 'train_v2.csv'
SUBMISSION_FILE = 'sample_submission_v2.csv'
TRAIN_PATH = 'train-jpg/'
TEST_PATH = 'test-jpg/'


print(MAIN)

x_train = []
x_test = []
y_train = []

logging.info('Reading TRAINING LABELS datasource')
df_labels = pd.read_csv(os.path.join(MAIN, TRAIN_LABELS))
logging.info('We read a training dataframe of shape' + str(df_labels.shape))

df_test = pd.read_csv(os.path.join(MAIN, SUBMISSION_FILE))


# flatten = lambda l: [item for sublist in l for item in sublist]
# print(flatten)
# labels = list(set(flatten([l.split(' ') for l in df_labels['tags'].values])))
# print(labels)
labels_list = []
for tag in df_labels.tags.values:
    labels = tag.split(' ')
    for label in labels:
        if label not in labels_list:
            labels_list.append(label)

# def plot_pictures(label, df_train, train_path):
#
#     images = df_train[df_train[label] == 1].image_name.values
#
#     fig , ax = plt.subplots(nrows=3, ncols=3, figsize=(8,8))
#     ax = ax.flatten()
#
#     for i in range(0,9):
#         f = random.choice(images)
#         img = Image.open(os.path.join(train_path, f + '.jpg'))
#         ax[i].imshow(img)
#         ax[i].set_xticks([])
#         ax[i].set_yticks([])
#         ax[i].set_title("{}s h:{}s w:{}s".format(f, img.height, img.width))
#     plt.tight_layout()




# print(plot_pictures('clear', df_labels, TRAIN_PATH))



# print(labels_list)

# labels = ['blow_down',
#           'bare_ground',
#           'conventional_mine',
#           'blooming',
#           'cultivation',
#           'artisinal_mine',
#           'haze',
#           'primary',
#           'slash_burn',
#           'habitation',
#           'clear',
#           'road',
#           'selective_logging',
#           'partly_cloudy',
#           'agriculture',
#           'water',
#           'cloudy']

# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

label_map = {'blow_down': 0,'bare_ground': 1,'conventional_mine': 2,'blooming': 3,'cultivation': 4,'artisinal_mine': 5,
            'haze': 6,'primary': 7,'slash_burn': 8,'habitation': 9,'clear': 10,'road': 11,'selective_logging': 12,
            'partly_cloudy': 13,'agriculture': 14, 'water': 15,'cloudy': 16}

for f, tags in tqdm(df_labels.values[:18000], miniters=1000):
    train_img = cv2.imread(os.path.join(MAIN, TRAIN_PATH) + '{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(train_img, (64, 64)))
    y_train.append(targets)

# print(targets)

for f, tags in tqdm(df_test.values[:18000], miniters=1000):
    img = cv2.imread(os.path.join(MAIN, TEST_PATH) + '{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (64, 64)))



y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.

print("x_test shape: " + str(x_test.shape))
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))


# prediction = np.random.rand(18000, 17)
# true_label = np.random.rand(18000, 17) > 0.5

# start = time.time()
# t1 = get_optimal_threshhold(true_label, prediction)
# print(time.time() - start)
# start = time.time()
#
# t2 = optimise_f2_thresholds(true_label, prediction)
# print(time.time() - start)

"""
    module takes 388.4562318325043 to run or 6.5 mins
"""

nfolds = 3

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train = []

kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

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

    kfold_weights_path = os.path.join('.weights/', 'weights_kfold_' + str(num_fold) + '.h5')

    # model = Sequential([
    #     BatchNormalization(input_shape=(64, 64, 3)),
    #     Conv2D(8, 1, 1, activation='relu'),
    #     Conv2D(16, 2, 2, activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(32, 3, 3, activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Conv2D(64, 3, 3, activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(256, activation='relu'),
    #     Dropout(0.5),
    #     Dense(17, activation='sigmoid')
    # ])
    model = Sequential()
    #  TODO: this is from the CV + Keras example
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))
    # model.add(Conv2D(8, 1, 1, activation='relu'))
    # model.add(Conv2D(16, 2, 2, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, 3, 3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(17, activation='sigmoid'))

    model.add(BatchNormalization(input_shape=(64, 64, 3)))
    model.add(BatchNormalization(input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    # epochs_arr = [20, 5, 5]
    # learn_rates = [0.001, 0.0001, 0.00001]
    #
    # for learn_rate, epochs in zip(learn_rates, epochs_arr):
    #     opt = optimizers.Adam(lr=learn_rate)
    #     model.compile(loss='binary_crossentropy',
    #                   # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    #                   optimizer=opt,
    #                   metrics=['accuracy'])
    #     callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #                  ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
    #
    #     model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
    #               batch_size=128, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=2, verbose=0, min_delta=1e-4),  # adding min_delta
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),  # new callback
    #     ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]




    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=2, epochs=1, #callbacks=[callbacks],
              shuffle=True)

    from keras.utils.vis_utils import plot_model

    plot_model(model, to_file='model.png')

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=128, verbose=2)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
    print("Optimizing prediction threshold")
    print(optimise_f2_thresholds(Y_valid, p_valid))

    p_train = model.predict(x_train, batch_size=128, verbose=2)
    yfull_train.append(p_train)

    p_test = model.predict(x_test, batch_size=128, verbose=2)
    yfull_test.append(p_test)





result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns=labels_list)

thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

# print(preds)
print("the size of preds list is " + str(len(preds)))

try:
    df_test['tags'] = preds
except Exception as err:
    print(err)
    pass

# print("the size of df_test is " + str(len(df_test)))
print("the shape of df_test is " + str(df_test.shape))

# df_test['tags'] = preds
# print(df_test)

print(time.time() - start_time)

