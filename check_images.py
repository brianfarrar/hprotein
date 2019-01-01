import cv2
import hprotein
from tqdm import tqdm
import pandas as pd
import numpy as np

LABEL_PATH = 'stage1_labels'
LABEL_LIST = 'train_combo.csv'
TRAIN_PATH = 'stage1_train_combo'

df = pd.read_csv(LABEL_PATH + '/' + LABEL_LIST)
print('Starting with {} examples'
      ''.format(len(df)))
example_list = df.values


COLORS = ['red','green', 'blue', 'yellow']

good_example_list = []
bad_image = False

for example in tqdm(example_list):
    for color in COLORS:
        fname = hprotein.get_image_fname(TRAIN_PATH, example[0], color)
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if image is None:
            bad_image = True

    if bad_image:
        bad_image = False
    else:
        good_example_list.append([example[0], example[1]])

print('There are {} now'.format(len(good_example_list)))
df = pd.DataFrame(np.asarray(good_example_list))
df.to_csv('new_train_combo.csv', index=False, header=['Id','Target'])