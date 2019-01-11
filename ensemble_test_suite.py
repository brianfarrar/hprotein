import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import os

from sklearn.metrics import f1_score

EVAL_FOLDER = 'eval'
eval_file_list = [f for f in os.listdir(EVAL_FOLDER) if f.startswith("eval")]


def make_label_array(labels):
    # set up a numpy array to receive the encoded label
    label_array = np.zeros(28)

    if isinstance(labels, str):

        # split the space separated multi-label into a list of individual labels
        split_label = labels.split(' ')

        # turn on the positive columns in the labels array
        for label in split_label:
            label_array[np.uint8(label)] = 1

    return label_array


for f in eval_file_list:

    df_eval = pd.read_csv(EVAL_FOLDER + '/' + f)
    df_gt = df_eval[['Id', 'Ground_Truth']].copy()
    df_pred = df_eval[['Id', 'Predictions']].copy()

    predictions = []
    ground_truth = []

    for i in range(len(df_eval)):
        p = make_label_array(df_pred['Predictions'][i])
        gt = make_label_array(df_gt['Ground_Truth'][i])
        predictions.append(p)
        ground_truth.append(gt)

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    #print(predictions.shape)
    #print(ground_truth.shape)
    #print(len(df_eval))

    score = f1_score(ground_truth, predictions, average='macro')
    print('{} -> Macro-F1: {}'.format(f, score))