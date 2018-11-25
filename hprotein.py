# imports
import logging
import distutils.util
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.lib.io import file_io

import keras
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split


# constants
COLORS = ['red','green', 'blue', 'yellow']
#SHAPE = (192, 192, 4)
SHAPE = (512, 512, 4)
THRESHOLD = 0.05
SEED = 42

# -------------------------------------------------------------
# Converts a text based "True" or "False" to a python bool
# -------------------------------------------------------------
def text_to_bool(text_bool):
    return bool(distutils.util.strtobool(text_bool))


# ------------------------------------------------
# copies a file to google cloud storage
# ------------------------------------------------
def copy_file_to_gcs(fname_in, fname_out):

    logging.info('Writing {} to gcs at {}...'.format(fname_in, fname_out))
    with file_io.FileIO(fname_in, mode='rb') as f_in:
        with file_io.FileIO(fname_out, mode='wb+') as f_out:
            f_out.write(f_in.read())


# -----------------------------------------
# get a list of unique specimen ids
# -----------------------------------------
def get_specimen_ids(path):
    # get a list of all the images
    file_list = file_io.list_directory(path)

    # truncate the file names to make a specimen id
    specimen_ids = [f[:36] for f in file_list]

    # eliminate duplicates
    specimen_ids = list(set(specimen_ids))

    return specimen_ids


# -----------------------------------------
# make filenames from the specimen id
# -----------------------------------------
def get_image_fname(path, specimen_id, color, lo_res=True):

    # construct filename
    if lo_res:
        fname = path + '/' + specimen_id + '_' + color + '.png'
    else:
        fname = path + '/' + specimen_id + '_' + color + '.tif'

    return fname


# ---------------------------------------
# Keras style run time data generator
# ---------------------------------------
class HproteinDataGenerator(keras.utils.Sequence):

    # ---------------------------------------------
    # Required function to initialize the class
    # ---------------------------------------------
    def __init__(self,
                 args,
                 path,
                 specimen_ids,
                 labels,
                 shape=SHAPE,
                 shuffle=False,
                 use_cache=False,
                 augment=False):

        self.args = args
        self.path = path
        self.specimen_ids = specimen_ids  # list of features
        self.labels = labels  # list of labels
        self.batch_size = args.batch_size  # batch size
        self.shape = shape  # shape of features
        self.shuffle = shuffle  # boolean for shuffle
        self.use_cache = use_cache  # boolean for use of cache
        self.augment = augment  # boolean for image augmentation

    # -------------------------------------------------------
    # Required function to determine the number of batches
    # -------------------------------------------------------
    def __len__(self):
        return int(np.ceil(len(self.specimen_ids) / float(self.batch_size)))

    # -------------------------------------------------------
    # Required function to get a batch
    # -------------------------------------------------------
    def __getitem__(self, index):

        # get the list of specimen ids for this batch
        specimen_ids = self.specimen_ids[self.batch_size * index:self.batch_size * (index + 1)]

        # create a zeroed out numpy array to load the batch into
        feature_batch = np.zeros((len(specimen_ids), self.shape[0], self.shape[1], self.shape[2]))

        # load a batch of labels
        label_batch = self.labels[self.batch_size * index:self.batch_size * (index + 1)]

        # load a batch of images
        if self.use_cache:
            print("Error: use_cache not implemented!")
        else:
            for i, specimen_id in enumerate(specimen_ids):
                feature_batch[i] = self.get_stacked_image(specimen_id)

        # augment images if desired
        if self.augment:
            print("Error: Image augmentation not implemented!")

        return feature_batch, label_batch

    # -----------------------------------------
    # get a single image
    # -----------------------------------------
    def get_single_image(self, specimen_id, color, lo_res=True):

        # get image file name
        fname = get_image_fname(self.path, specimen_id, color, lo_res)

        # read image as a 1-channel image
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.shape[0], self.shape[1]))

        return image

    # -----------------------------------------
    # get a stacked (4-channel) image
    # -----------------------------------------
    def get_stacked_image(self, specimen_id, lo_res=True):

        # create a numpy array to place the 1-channel images into
        image = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.uint8)

        for n, color in enumerate(COLORS):
            # get a single image
            i = self.get_single_image(specimen_id, color, lo_res)

            # store it a channel
            image[:, :, n] = i

        return image


# -----------------------------------------------------------
# get the available specimen ids and corresponding labels
# -----------------------------------------------------------
def get_data(train_path, label_path):

    # get the list of specimen ids
    specimen_ids = get_specimen_ids(train_path)

    # get the labels for all specimen_ids
    label_data = pd.read_csv(label_path)

    # get the subset of labels that match the specimen images that are on TRAIN_PATH
    labels_subset = label_data.loc[label_data['Id'].isin(specimen_ids)]

    #
    # convert labels to trainer format
    #

    # set up the list that will contain the list of encoded labels for each specimen id
    labels = []

    # loop through each specimen_id
    for specimen_id in specimen_ids:

        # split the space separated multi-label into a list of individual labels
        split_labels = (labels_subset.loc[labels_subset['Id'] == specimen_id])['Target'].str.split(' ')

        # set up a numpy array to receive the encoded label
        l = np.zeros(28, dtype=np.uint8)

        # turn on the positive columns in the labels array
        for label in split_labels:
            l[np.uint8(label)] = 1

        labels.append(l)

    return np.array(specimen_ids), np.array(labels)


# -----------------------------------------------------------
# get the specimen ids to predict
# -----------------------------------------------------------
def get_predict_data(test_path, output_path):
    # get the list of specimen ids for which there are images
    specimen_ids = get_specimen_ids(test_path)

    # get the list of submission specimen ids required
    submit_data = pd.read_csv(output_path + '/sample_submission.csv')

    # get the subset of labels that match the specimen images that are on TEST_PATH
    submit_subset = submit_data.loc[submit_data['Id'].isin(specimen_ids)]

    # set up the list that will contain the list of encoded labels for each specimen id
    predicted_labels = np.zeros((len(specimen_ids), 28), dtype=np.uint8)

    return np.array(specimen_ids), predicted_labels


# -----------------------------
# get train/test split
# -----------------------------
def get_train_test_split(args, test_size=3072):

    logging.info('Loading datasets from {} and {} ...'.format(args.train_folder, args.label_folder))
    specimen_ids, labels = get_data(args.train_folder, args.label_folder)

    train_set_sids, val_set_sids, \
    train_set_lbls, val_set_lbls = train_test_split(specimen_ids, labels, test_size=test_size, random_state=SEED)
    logging.info('Created train|test split of {}|{}'.format(len(train_set_lbls), len(val_set_lbls)))

    return train_set_sids, val_set_sids, train_set_lbls, val_set_lbls


# --------------------------------
# calculate the f1 statistic
# --------------------------------
def f1(y_true, y_pred):

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


# --------------------------------
# calculate the f1 loss
# --------------------------------
def f1_loss(y_true, y_pred):

    f = f1(y_true, y_pred)

    return 1 - K.mean(f)


# ------------------------------
# create the model
# ------------------------------
def create_model(input_shape):

    dropRate = 0.25

    init = Input(input_shape)
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    c1 = Conv2D(16, (3, 3), padding='same')(x)
    c1 = ReLU()(c1)
    c2 = Conv2D(16, (5, 5), padding='same')(x)
    c2 = ReLU()(c2)
    c3 = Conv2D(16, (7, 7), padding='same')(x)
    c3 = ReLU()(c3)
    c4 = Conv2D(16, (1, 1), padding='same')(x)
    c4 = ReLU()(c4)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    # x = Conv2D(256, (1, 1), activation='relu')(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(28)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)

    model = Model(init, x)

    return model


def get_max_fscore_matrix(val_predictions, val_labels):

    # get a range between 0 and 1 by 1000ths
    rng = np.arange(0, 1, 0.001)

    # set up an array to catch individual fscores for each class
    fscores = np.zeros((rng.shape[0], 28))

    # loop through each prediction above the a threshold and calculate the fscore
    for j,k in enumerate(rng):
        for i in range(28):
            p = np.array(val_predictions[:,i]>k, dtype=np.int8)
            score = f1_score(val_labels[:,i], p, average='binary')
            fscores[j,i] = score

    # log results for inspection
    logging.info('Individual F1-scores for each class:')
    logging.info(np.max(fscores, axis=0))
    logging.info('Macro F1-score CV ='.format(np.mean(np.max(fscores, axis=0))))

    # Make a matrix that will hold the best threshold for each class to maximize Fscore
    max_fscore_thresholds = np.empty(28, dtype=np.float16)
    for i in range(28):
        max_fscore_thresholds[i] = rng[np.where(fscores[:,i] == np.max(fscores[:,i]))[0][0]]

    logging.info('Probability threshold maximizing CV F1-score for each class:')
    logging.info(max_fscore_thresholds)

    return max_fscore_thresholds