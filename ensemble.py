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

    parser.add_argument('--ensemble_csv', dest='ensemble_csv', default='ensemble_models',
                        help='Folder containing the models to ensemble')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='Number of examples per batch')

    parser.add_argument('--use_class_weights', dest='use_class_weights', default='False',
                        help='Text boolean to decide whether to use class weights in training')

    parser.add_argument('--gpu_count', dest='gpu_count', default=1, type=int,
                        help='Number of gpus to use for training')

    parser.add_argument('--val_csv', dest='val_csv', default='val_set.csv',
                        help='CSV with the validation set info.')

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

    ensemble_predictions(args)


# --------------------------------------
# ensembles predictions into a single
# --------------------------------------
def ensemble_predictions(args):

    # get predict data
    logging.info('Reading predict test set from {}...'.format(args.predict_folder))
    predict_set_sids, predict_set_lbls = hprotein.get_data(args.submission_folder, args.submission_list, mode='test')
    predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids, predict_set_lbls,
                                                       model_name=args.model_name)

    # get model list
    model_list = hprotein.get_model_list(args)

    # create an empty array to catch the predictions
    if args.gpu_count > 1:
        # keras multi gpu models require all batches in the prediction run to be full, so we pad for the last batch
        predictions = np.zeros((len(model_list), predict_set_sids.shape[0] + predict_generator.last_batch_padding, 28))
    else:
        predictions = np.zeros((len(model_list), predict_set_sids.shape[0], 28))

    # read in the list of samples in the correct order to submit
    submit = pd.read_csv('{}/{}'.format(args.submission_folder, args.submission_list))

    # loop through each model and generate scores
    for j, m in enumerate(model_list):

        # get the best model and related threshold matrix
        model, max_thresholds_matrix = hprotein.get_best_model(args.model_folder, m[0])

        if args.gpu_count > 1:
            # compile model with desired loss function
            if m[1] == 'binary_crossentropy':
                model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
            elif m[1] == 'focal_loss':
                model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=1e-4),
                                    metrics=['accuracy', hprotein.f1])

            model.summary()

        # get the predictions
        logging.info('Making predictions...')
        for i in tqdm(range(len(predict_generator))):
            images, labels = predict_generator[i]

            # if the last batch is not full append blank rows
            if images.shape[0] < predict_generator.batch_size:
                if args.gpu_count > 1:
                    blank_rows = np.zeros((predict_generator.last_batch_padding,
                                           predict_generator.shape[0],
                                           predict_generator.shape[1],
                                           predict_generator.shape[2]))
                    images = np.append(images, blank_rows, axis=0)

            score = model.predict(images)
            predictions[j, i * predict_generator.batch_size : ((i * predict_generator.batch_size) + score.shape[0])] = score

    final_predictions = np.mean(predictions, axis=0)

    # drop the blank rows
    if args.gpu_count > 1:
        # keras multigpu models require all batches in the prediction run to be full, so we drop the padded predictions
        if images.shape[0] < predict_generator.batch_size:
            final_predictions = predictions[:predictions.shape[1] - predict_generator.last_batch_padding]

    # thresholds

    # convert the predictions into the submission file format
    logging.info('Converting to submission format...')
    prediction_str = []
    for row in tqdm(range(submit.shape[0])):
        str_label = ''
        for col in range(final_predictions.shape[1]):
            if final_predictions[row, col] < max_thresholds_matrix[col]:
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction_str.append(str_label.strip())

    # add column to pandas dataframe for submission
    submit['Predicted'] = np.array(prediction_str)

    # write out the csv
    submit.to_csv('{}/submit_{}.csv'.format(args.submission_folder, args.model_label), index=False)

    # copy the submission file to gcs
    hprotein.copy_file_to_gcs('{}/submit_{}.csv'.format(args.submission_folder, args.model_label),
                              'gs://hprotein/submission/submit_{}'.format(args.model_label))

    logging.info('Model: {} prediction run complete!'.format(args.model_label))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
