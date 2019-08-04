import time
import logging

from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from src.utillities.base_task import BaseTask
from src.common import fcts
fcts.init_logger('~/PycharmProjects/AmazonKaggle-MlCapstone/Logs/', 'AmazonProccessor')


class AmazonProcessor(BaseTask):
    """
    This class will be served as the main function to train the classifier, and build the submission file (csv).
    """

    @staticmethod
    def log_title(title):
        logging.info('************************************')
        logging.info('*********' + str(title) + '*********')
        logging.info('************************************')

    @staticmethod
    def log_subtitle(title):
        logging.info('*********' + str(title) + '*********')


    def iteration(self):
        self.defaults = BaseTask.defaults
        self.label_map = \
                {'blow_down'        : 0,
                 'bare_ground'      : 1,
                 'conventional_mine': 2,
                 'blooming'         : 3,
                 'cultivation'      : 4,
                 'artisinal_mine'   : 5,
                 'haze'             : 6,
                 'primary'          : 7,
                 'slash_burn'       : 8,
                 'habitation'       : 9,
                 'clear'            : 10,
                 'road'             : 11,
                 'selective_logging': 12,
                 'partly_cloudy'    : 13,
                 'agriculture'      : 14,
                 'water'            : 15,
                 'cloudy'           : 16}

        self.result = True

        # define the
        self.x_train = []
        self.y_train = []
        self.x_test = []

        self.start_time = time.time()


        self.image_size = self.resize_image[0]
        self.df_labels = self._get_labels()
        self.submission_file = self._get_test()

        return self.result

    def _get_labels(self):
        self.log_subtitle('Reading training labels')
        self.df_labels = pd.read_csv(os.path.join(self.main, self.train_labels))
        logging.info('Successfully read training labels : ' + str(self.df_labels.shape))
        return self.df_labels

    def _get_test(self):
        self.log_subtitle('Reading testing data')
        self.submission_file = pd.read_csv(os.path.join(self.main, self.submission_file))
        logging.info('Successfully read submission data : ' + str(self.submission_file.shape))
        return self.submission_file

    def process_training_data(self):
        """Function to process the training data and return x_train, y_train"""
        for f, tags in tqdm(self.df_labels.values[:18000], miniters=1000):
            train_img = cv2.imread(os.path.join(self.main, self.train_path) + '{}.jpg'.format(f))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[self.label_map[t]] = 1
            # train_img = cv2.resize(train_img, self.resize_image)
            self.x_train.append(cv2.resize(train_img, (64, 64)))
            self.y_train.append(targets)

            flipped_img = cv2.flip(train_img, 1)
            rows, cols, channel = train_img.shape

            # regular image
            self.x_train.append(train_img)
            self.y_train.append(targets)

            # flipped image
            self.x_train.append(flipped_img)
            self.y_train.append(targets)

            for rot_deg in [90, 180, 270]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_deg, 1)
                dst = cv2.warpAffine(train_img, M, (cols, rows))
                self.x_train.append(dst)
                self.y_train.append(targets)

                dst = cv2.warpAffine(flipped_img, M, (cols, rows))
                self.x_train.append(dst)
                self.y_train.append(targets)

            # self.y_train = np.array(self.y_train, np.uint8)
            # self.x_train = np.array(self.x_train, np.uint8)

        self.y_train = np.array(self.y_train, np.uint8)
        self.x_train = np.array(self.x_train, np.float32) / 255.

        return self.build_training_classifier(self.x_train, self.y_train)

    def build_training_classifier(self, x_train, y_train):

        model = self.amazon_sequential_custom_build()

        logging.info('Building training model')
        kf = KFold(len(y_train), n_folds=self.nfolds, shuffle=True, random_state=1)

        num_fold = 0
        for train_index, test_index in kf:
            start_time_model_fitting = time.time()

            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_valid = x_train[test_index]
            Y_valid = y_train[test_index]

            num_fold += 1
            logging.info('KFold {} out of {}'.format(num_fold, self.nfolds))
            logging.info('Split train size: ', len(X_train), len(Y_train))
            logging.info('Split valid size: ', len(X_valid), len(Y_valid))

            kfold_weights_path = os.path.join('.weights/', 'weights_kfold_' + str(num_fold) + '.h5')

            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
            # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=0, min_delta=1e-4),  # adding min_delta
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
                # new callback
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

            model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                      batch_size=128, verbose=2, epochs=10, callbacks=[callbacks],
                      shuffle=True)

            from keras.utils import plot_model
            plot_model(model, to_file='model.png')

            if os.path.isfile(kfold_weights_path):
                model.load_weights(kfold_weights_path)

            # p_valid = model.predict(X_valid, batch_size=self.batch_size, verbose=2)
            p_test = model.predict(self.x_test, batch_size=self.batch_size, verbose=2)
            logging.info(fbeta_score(Y_valid, np.array(p_test) > 0.18, beta=2, average='samples'))


    def predict(self, x_test, nfolds=3, batch_size=128):
        self.log_subtitle('Prediction')
        model = self.amazon_sequential_custom_build()
        y_test = []

        for num_fold in range(1, self.nfolds + 1):
            weight_path = os.path.join('' + 'weights_kfold_' + str(num_fold) + '.h5')

            if os.path.isfile(weight_path):
                model.load_weights(weight_path)

            p_test = model.predict(x_test, batch_size=batch_size, verbose=2)
            y_test.append(p_test)

        result = np.array(y_test[0])
        for i in range(1, self.nfolds):
            result += np.array(y_test[i])
        result /= nfolds

        return result

    def amazon_sequential_custom_build(self, input_shape=(128, 128), weight_path=None):
        """
        :param input_shape:
        :return: Keras Object
        """
        custom_model = Sequential()
        custom_model.add(BatchNormalization(input_shape=input_shape))
        custom_model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        custom_model.add(Conv2D(32, (3, 3), activation='relu'))
        custom_model.add(MaxPooling2D(pool_size=(2, 2)))
        custom_model.add(Dropout(0.25))

        custom_model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        custom_model.add(Conv2D(64, (3, 3), activation='relu'))
        custom_model.add((MaxPooling2D(pool_size=(2, 2))))
        custom_model.add(Dropout(0.25))

        custom_model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        custom_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        custom_model.add(MaxPooling2D(pool_size=(2, 2)))
        custom_model.add(Dropout(0.25))

        custom_model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        custom_model.add(Conv2D(256, (3, 3), activation='relu'))
        custom_model.add(MaxPooling2D(pool_size=(2, 2)))
        custom_model.add(Dropout(0.25))

        custom_model.add(Flatten())
        custom_model.add(Dense(512, activation='relu'))
        custom_model.add(BatchNormalization())
        custom_model.add(Dropout(0.5))
        custom_model.add(Dense(17, activation='sigmoid'))
        if (weight_path != None):
            if os.path.isfile(weight_path):
                custom_model.load_weights(weight_path)
        return custom_model

    def build_prediction(self):

        x_test = []

        for f, tags in tqdm(self.submission_file.values, miniters=1000):
            test_img = cv2.imread(os.path.join(self.main, self.test_path) + '{}.jpg'.format(f))
            x_test.append(cv2.resize(test_img, (64, 64)))

        x_test = np.array(x_test, np.float32) / 255.

        result = self.predict(x_test)

        labels_list = []
        for tag in self.df_labels.tags.values:
            labels = tag.split(' ')
            for label in labels:
                if label not in labels_list:
                    labels_list.append(label)

        result = pd.DataFrame(result, columns=labels_list)

        thres = {'blow_down': 0.2,
                 'bare_ground': 0.138,
                 'conventional_mine': 0.1,
                 'blooming': 0.168,
                 'cultivation': 0.204,
                 'artisinal_mine': 0.114,
                 'haze': 0.204,
                 'primary': 0.204,
                 'slash_burn': 0.38,
                 'habitation': 0.17,
                 'clear': 0.13,
                 'road': 0.156,
                 'selective_logging': 0.154,
                 'partly_cloudy': 0.112,
                 'agriculture': 0.164,
                 'water': 0.182,
                 'cloudy': 0.076}

        preds = []
        for i in tqdm(range(result.shape[0]), miniters=1000):
            res = result.ix[[i]]
            pred_tag = []
            for k, v in thres.items():
                if res[k][i] >= v:
                    pred_tag.append(k)
            preds.append(' '.join(pred_tag))

        if self.write_submission_file:
            df_test = pd.DataFrame()
            df_test['tags'] = preds
            df_test.to_csv('sub.csv', index=False)
