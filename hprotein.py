# imports
import logging
import distutils.util
import cv2
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.lib.io import file_io

import keras
from keras import backend as K
from keras.activations import selu
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, \
                         Concatenate, ReLU, GlobalAveragePooling2D

from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.layers.noise import AlphaDropout
from keras.models import load_model

#from sklearn.metrics import f1_score as off1
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore",
                        message="F-score is ill-defined and being set to 0.0 due to no true samples.")


import imgaug as ia
from imgaug import augmenters as iaa

# constants
COLORS = ['red','green', 'blue', 'yellow']
IMAGE_SIZE = 512
THRESHOLD = 0.5
SEED = 42

from tensorflow import set_random_seed
set_random_seed(SEED)

# ------------------------------------------------
# mini train set for testing purposes
# ------------------------------------------------
mini_train_set = ['000c99ba-bba4-11e8-b2b9-ac1f6b6435d0',
                  '001bcdd2-bbb2-11e8-b2ba-ac1f6b6435d0',
                  '0020af02-bbba-11e8-b2ba-ac1f6b6435d0',
                  'fb4c1fac-bbaa-11e8-b2ba-ac1f6b6435d0',
                  'fc84a97c-bbad-11e8-b2ba-ac1f6b6435d0',
                  'fea6e496-bbbb-11e8-b2ba-ac1f6b6435d0',
                  'fffe0ffe-bbc0-11e8-b2bb-ac1f6b6435d0'
]

# ------------------------------------------------
# mini validation set for testing purposes
# ------------------------------------------------
mini_validate_set = ['001838f8-bbca-11e8-b2bc-ac1f6b6435d0',
                     '002daad6-bbc9-11e8-b2bc-ac1f6b6435d0',
                     'ffeae6f0-bbc9-11e8-b2bc-ac1f6b6435d0'
]

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
def get_specimen_ids(path_name, list_name):

    df = pd.read_csv(path_name + '/' + list_name)
    s = df.values
    specimen_ids = s[:, 0]

    return specimen_ids


# -----------------------------------------
# make file names from the specimen id
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
                 model_name='gap_net_bn_relu',
                 shuffle=False,
                 augment=False):

        self.args = args
        self.path = path
        self.specimen_ids = specimen_ids  # list of features
        self.labels = labels  # list of labels
        self.batch_size = args.batch_size  # batch size
        self.model_name = model_name

        # get the number of examples to generate
        example_count = len(self.specimen_ids)

        # calculate the number of batches
        self.batch_count = int(np.ceil(example_count / float(self.batch_size)))

        # get the size of the last batch
        last_batch_size = example_count - ((self.batch_count - 1) * self.batch_size)

        # set the amount to pad the last batch
        self.last_batch_padding = self.batch_size - last_batch_size

        # shape of features
        if model_name == 'InceptionV2Resnet':
            self.shape = (299, 299, 3)
        else:
            self.shape = (IMAGE_SIZE, IMAGE_SIZE, 4)

        self.shuffle = shuffle  # boolean for shuffle
        self.augment = augment  # boolean for image augmentation

        ia.seed(SEED)

    # -------------------------------------------------------
    # Required function to determine the number of batches
    # -------------------------------------------------------
    def __len__(self):
        return self.batch_count

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

        # load the batch with images
        for i, specimen_id in enumerate(specimen_ids):
            feature_batch[i] = self.get_stacked_image(specimen_id)

        return feature_batch, label_batch

    # -----------------------------------------
    # get a single image
    # -----------------------------------------
    def get_single_image(self, specimen_id, color, lo_res=True):

        # get image file name
        fname = get_image_fname(self.path, specimen_id, color, lo_res)

        # read image as a 1-channel image
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        if self.model_name == 'InceptionV2Resnet':
            image = cv2.resize(image, (self.shape[0], self.shape[1]))

        return image

    # -----------------------------------------
    # get a stacked (3 or 4-channel) image
    # -----------------------------------------
    def get_stacked_image(self, specimen_id, lo_res=True):

        # create a numpy array to place the 1-channel images into
        image = np.zeros((self.shape))

        for n in range(self.shape[2]):

            # get a single image
            i = self.get_single_image(specimen_id, COLORS[n], lo_res)

            # store it a channel
            image[:, :, n] = i

        # augment images if desired
        if self.augment:
            seq = iaa.Sequential([
                iaa.Fliplr(1),  # horizontal flips
                iaa.Flipud(1),  # horizontal flips
                iaa.Crop(percent=(0, 0.1)),  # random crops
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
            ], random_order=True)  # apply augmenters in random order

            # augment 25% of the time
            if np.random.uniform() > 0.75:
                image = seq.augment_image(image)

        image = np.divide(image, 255.)

        return image


# -----------------------------------------------------------
# get the available specimen ids and corresponding labels
# -----------------------------------------------------------
def get_data(path_name, list_name, mode='train', filter_ids=[]):

    # get the list of specimen ids
    specimen_ids = get_specimen_ids(path_name, list_name)

    if mode is 'train':
        specimen_ids = [specimen_id for specimen_id in specimen_ids if specimen_id not in filter_ids]
    elif mode in ['validate', 'mini-train', 'mini-validate']:
        specimen_ids = [item for item in specimen_ids if item in filter_ids]
    elif mode is 'test':
        pass
    else:
        logging.warning('Invalid mode {} specified'.format(mode))

    # if the mode is 'test' then create an empty label set, otherwise convert labels to trainer format
    if mode is 'test':

        # set up an array to receive predicted labels
        labels = np.ones((len(specimen_ids), 28))

    else:

        # set up the list that will contain the list of decoded labels for each specimen id
        labels = []

        # read in the ground truth labels
        df_labels = pd.read_csv(path_name + '/' + list_name)

        # loop through each specimen_id
        logging.info('Decoding train/validate labels...')
        for specimen_id in tqdm(specimen_ids):

            # split the space separated multi-label into a list of individual labels
            split_labels = (df_labels.loc[df_labels['Id'] == specimen_id])['Target'].str.split(' ')

            # set up a numpy array to receive the encoded label
            l = np.zeros(28)

            # turn on the positive columns in the labels array
            for label in split_labels:
                l[np.uint8(label)] = 1

            labels.append(l)

    return np.array(specimen_ids), np.array(labels)


# --------------------------------
# calculate the f1 statistic
# --------------------------------
def f1(y_true, y_pred):

    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
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

# --------------------------------
# focal loss
# --------------------------------
def focal_loss(y_true, y_pred, gamma=2):

    # transform back to logits
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    input = tf.cast(y_pred, tf.float32)

    max_val = K.clip(-input, 0, 1)
    loss = input - input * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (y_true * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))


# ------------------------------
# create the model
# ------------------------------
def create_model(model_name='basic_cnn'):

    # determine the input shape
    if model_name == 'InceptionV2Resnet':
        input_shape = (299, 299, 3)
    else:
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 4)

    init = Input(input_shape)

    if model_name == 'basic_cnn':

        drop_rate = 0.25

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
        x = Dropout(drop_rate)(x)
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
        x = Dropout(drop_rate)(x)
        x = Conv2D(32, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(drop_rate)(x)
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(drop_rate)(x)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(drop_rate)(x)
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

    elif model_name == 'gap_net_selu':

        drop_rate = 0.25

        x = Conv2D(32, (3, 3), strides=(2, 2), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(init)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        gap_input1 = AlphaDropout(drop_rate)(x)

        x = Conv2D(64, (3, 3), strides=(2, 2), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        gap_input2 = AlphaDropout(drop_rate)(x)

        x = Conv2D(128, (3, 3), strides=(1, 1), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        gap_input3 = AlphaDropout(drop_rate)(x)

        gap1 = GlobalAveragePooling2D()(gap_input1)
        gap2 = GlobalAveragePooling2D()(gap_input2)
        gap3 = GlobalAveragePooling2D()(gap_input3)

        x = Concatenate()([gap1, gap2, gap3])

        x = Dense(256, activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = AlphaDropout(drop_rate)(x)
        x = Dense(256, activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = AlphaDropout(drop_rate)(x)
        x = Dense(28, activation=selu, kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
        x = Activation('sigmoid')(x)

    elif model_name == 'gap_net_bn_relu':

        drop_rate = 0.25

        x = BatchNormalization(axis=-1)(init)
        x = Conv2D(32, (3, 3))(x)  # , strides=(2,2))(x)
        x = ReLU()(x)

        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        ginp1 = Dropout(drop_rate)(x)

        x = BatchNormalization(axis=-1)(ginp1)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)

        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        ginp2 = Dropout(drop_rate)(x)

        x = BatchNormalization(axis=-1)(ginp2)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        ginp3 = Dropout(drop_rate)(x)

        gap1 = GlobalAveragePooling2D()(ginp1)
        gap2 = GlobalAveragePooling2D()(ginp2)
        gap3 = GlobalAveragePooling2D()(ginp3)

        x = Concatenate()([gap1, gap2, gap3])

        x = BatchNormalization(axis=-1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(drop_rate)(x)

        x = BatchNormalization(axis=-1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)

        x = Dense(28)(x)
        x = Activation('sigmoid')(x)

    elif model_name == 'InceptionV2Resnet':

        drop_rate = 0.25

        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

        x = BatchNormalization(axis=-1)(init)
        x = base_model(x)

        x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(28)(x)
        x = Activation('sigmoid')(x)
    else:
        logging.info('Bad model name: {}'.format(model_name))

    model = Model(init, x)

    return model


# ----------------------------------------------------
# Gets a matrix of thresholds that maximizes fscore
# ----------------------------------------------------
def get_max_fscore_matrix(model, val_generator, save_eval=False):

    # create empty arrays to receive the predictions and labels
    val_predictions = np.empty((0, 28))
    val_labels = np.empty((0, 28))

    # loop through the validation data and make predictions
    logging.info('Getting predictions...')
    for i in tqdm(range(len(val_generator))):
        image, label = val_generator[i]
        scores = model.predict(image)
        val_predictions = np.append(val_predictions, scores, axis=0)
        val_labels = np.append(val_labels, label, axis=0)

    # get a range between 0 and 1 by 1000ths
    rng = np.arange(0, 1, 0.001)

    # set up an array to catch individual fscores for each class
    fscores = np.zeros((rng.shape[0], 28))

    # loop through each prediction above the threshold and calculate the fscore
    logging.info('Calculating f-scores at a range of thresholds...')
    for j,k in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(val_predictions[:,i]>k, dtype=np.int8)
            score = f1_score(val_labels[:,i], p, average='binary')
            fscores[j,i] = score


    # Make a matrix that will hold the best threshold for each class to maximize Fscore
    max_fscore_thresholds = np.empty(28)
    for i in range(28):
        max_fscore_thresholds[i] = rng[np.where(fscores[:,i] == np.max(fscores[:,i]))[0][0]]

    macro_f1 = np.mean(np.max(fscores, axis=0))

    logging.info('Probability threshold maximizing F1-score for each class:')
    logging.info(max_fscore_thresholds)
    logging.info('Macro F1 Score -> {}'.format(macro_f1))


    # write out the eval csv
    if save_eval:
        # write_eval_csv(args, val_specimen_ids, val_predictions, max_fscore_thresholds)
        pass

    return max_fscore_thresholds, macro_f1


def xget_max_fscore_matrix(mdl, fullValGen, save_eval=False):
    lastFullValPred = np.empty((0, 28))
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(len(fullValGen))):
        im, lbl = fullValGen[i]
        scores = mdl.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
    print(lastFullValPred.shape, lastFullValLabels.shape)

    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            # scoref1 = K.eval(f1_score(fullValLabels[:,i], p, average='binary'))
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1

    print(np.max(f1s, axis=0))
    print(np.mean(np.max(f1s, axis=0)))

    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
    # print('Choosing threshold: ', T, ', validation F1-score: ', max(f1s))
    print(T)

    return T, np.mean(np.max(f1s, axis=0))


# ----------------------------------------------------
# Returns the best model and fscore matrix from disk
# ----------------------------------------------------
def get_best_model(args):

    final_model = None
    max_thresholds_matrix = None

    # Get thresholds
    logging.info('Getting correct model and thresholds...')
    if os.path.isfile('{}/{}_thresh.npy'.format(args.model_folder, args.model_label)):

        # load model
        logging.info('Loading model {}...'.format(args.model_label))
        final_model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                                 custom_objects={'f1': f1, 'focal_loss': focal_loss})

        # load thresholds
        max_thresholds_matrix = np.load('{}/{}_thresh.npy'.format(args.model_folder, args.model_label))

    elif os.path.isfile('{}/{}_fine_tune_thresh.npy'.format(args.model_folder, args.model_label)):

        # load model
        logging.info('Loading model {}_fine_tune...'.format(args.model_label))
        final_model = load_model('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                                 custom_objects={'f1': f1, 'f1_loss': f1_loss, 'focal_loss': focal_loss})

        # load thresholds
        max_thresholds_matrix = np.load('{}/{}_fine_tune_thresh.npy'.format(args.model_folder, args.model_label))
    else:
        logging.warning("Can't find model file {}/{}.model or {}/{}_fine_tune.model".format(args.model_folder,
                                                                                            args.model_label,
                                                                                            args.model_folder,
                                                                                            args.model_label))

    if max_thresholds_matrix is not None:
        logging.info('Using the following thresholds:')
        logging.info(max_thresholds_matrix)

    return final_model, max_thresholds_matrix


# ----------------------------------------------
# writes out a submission file
# ----------------------------------------------
def write_submission_csv(args, predict_set_sids, predictions, last_batch_padding, max_thresholds_matrix):

    # get the list of submission specimen ids required
    submit_data = pd.read_csv(args.submission_folder + '/sample_submission.csv')

    # get the subset of labels that match the specimen images that are on TEST_PATH
    submit_data = submit_data.loc[submit_data['Id'].isin(predict_set_sids)]
    tmp_sid_list = submit_data['Id'].values

    logging.info('Reformatting predictions and generating submission format...')

    # set up a list to receive the predictions in string form
    prediction_str = []

    # eliminate padding from end of prediction
    #predictions = predictions[:predictions.shape[0] - last_batch_padding, :]

    # loop through predictions and generate the prediction string
    for i in tqdm(range(submit_data.shape[0])):
        submit_str = ' '
        for j in range(predictions.shape[1]):
            if predictions[i, j] < max_thresholds_matrix[j]:
                submit_str += ''
            else:
                submit_str += str(j) + ' '

        prediction_str.append(submit_str.strip())

    submit_data['Predicted'] = np.array(prediction_str)

    submit_data.to_csv(args.submission_folder + '/submit_{}.csv'.format(args.model_label), index=False)


def write_eval_csv(args, val_specimen_ids, val_predictions, max_fscore_thresholds):

    # get the labels for all specimen_ids
    label_data = pd.read_csv(args.label_folder)

    # get the subset of labels that match the specimen images that are on TRAIN_PATH
    labels_subset = label_data.loc[label_data['Id'].isin(val_specimen_ids)]

    # set up a list to receive the predictions in string form
    val_predictions_str = []

    # loop through predictions and generate the prediction string
    for i in range(val_predictions.shape[0]):
        logging.info('Writing eval prediction #{} for specimen_id: {}'.format(i + 1, val_specimen_ids[i]))
        submit_str = ' '
        for j in range(val_predictions.shape[1]):
            if val_predictions[i, j] >= max_fscore_thresholds[j]:
                submit_str += str(j) + ' '

        val_predictions_str.append(submit_str.strip())

    # create dataframe and save to csv
    eval_output = pd.DataFrame()
    eval_output['Id'] = val_specimen_ids
    eval_output['Ground_Truth'] = label_data['Target']
    eval_output['Predictions'] = np.array(val_predictions_str)

    eval_output.to_csv(args.submission_folder + '/eval_{}.csv'.format(args.model_name), index=False)

# ----------------------------------------------------
# learning rate schedule for LearningRateSchedule
# ----------------------------------------------------
def lr_decay_schedule(change_point=15):

    def schedule(epoch):

        if epoch < change_point:
            lr = 1e-3
        else:
            lr = 1e-4

        logging.info('Current learning rate is {}'.format(lr))

        return lr

    return schedule
