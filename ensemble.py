import logging
import argparse
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
import hprotein

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model

# -------------------------------------------------------------
# Runs the model.  This is the main runner for this package.
# -------------------------------------------------------------
def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--ensemble_name', dest='ensemble_name', default='ensemble',
                        help='Network to run.')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='Number of examples per batch')

    parser.add_argument('--use_class_weights', dest='use_class_weights', default='False',
                        help='Text boolean to decide whether to use class weights in training')

    parser.add_argument('--model_name', dest='model_name', default='gap_net_bn_relu',
                        help='Network to run.')

    parser.add_argument('--model_label', dest='model_label', default='base',
                        help='Model to run.')

    parser.add_argument('--model_folder', dest='model_folder', default='./',
                        help='Folder to store the model checkpoints')

    parser.add_argument('--label_folder', dest='label_folder', default='./stage1_labels',
                        help='Folder that contains the train label csv')

    parser.add_argument('--label_list', dest='label_list', default='train.csv',
                        help='Folder that contains the train label csv')

    parser.add_argument('--train_folder', dest='train_folder', default='./stage1_data',
                        help='Folder that contains the train data files')

    parser.add_argument('--predict_folder', dest='predict_folder', default='./stage1_predict',
                        help='Folder that contains the data files to predict')

    parser.add_argument('--submission_folder', dest='submission_folder', default='./stage1_submit',
                        help='Folder for submission csv')

    parser.add_argument('--submission_list', dest='submission_list', default='sample_submission.csv',
                        help='File with submission list')


    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # Create a unique ensemble name
    unique_id = str(uuid.uuid4())[-6:]
    logging.info('Ensemble -> {}'.format(unique_id))
    args.ensemble_name = '{}_{}'.format(args.ensemble_name, unique_id)

