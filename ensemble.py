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

    # get validation data generator
    validation_set, val_generator = hprotein.get_val_generator(args)

    # get model list
    model_list = hprotein.get_model_list(args)

    #
    # Use validation data to calculate thresholds
    #

    # create an empty array to catch the validation predictions
    if args.gpu_count > 1:
        # keras multi gpu models require all batches in the prediction run to be full, so we pad for the last batch
        val_predictions = np.zeros((len(model_list), len(validation_set) + val_generator.last_batch_padding, 28))
        val_labels = np.zeros((len(model_list), len(validation_set) + val_generator.last_batch_padding, 28))
    else:
        val_predictions = np.zeros((len(model_list), len(validation_set), 28))
        val_labels = np.zeros((len(model_list), len(validation_set), 28))

    # loop through the validation data and make predictions
    logging.info('Calculating ensembled thresholds...')
    for j, m in enumerate(model_list):

        model, _ = hprotein.get_best_model(args.model_folder, m[0])
        model.summary()

        if args.gpu_count > 1:
            # compile model with desired loss function
            if m[1] == 'binary_crossentropy':
                model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
            elif m[1] == 'focal_loss':
                model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])

        # get validation data generator
        # get it each loop because the generator creates
        # different data shapes depending upon the model
        args.model_name = m[2]
        validation_set, val_generator = hprotein.get_val_generator(args)

        # get the validation predictions
        logging.info('Making validation predictions...')
        for i in tqdm(range(len(val_generator))):
            images, labels = val_generator[i]

            # if the last batch is not full append blank rows
            if images.shape[0] < val_generator.batch_size:
                if args.gpu_count > 1:
                    blank_rows = np.zeros((val_generator.last_batch_padding,
                                           val_generator.shape[0],
                                           val_generator.shape[1],
                                           val_generator.shape[2]))
                    images = np.append(images, blank_rows, axis=0)

            score = model.predict(images)
            val_predictions[j, i * val_generator.batch_size : ((i * val_generator.batch_size) + score.shape[0])] = score
            val_labels[j, i * val_generator.batch_size : ((i * val_generator.batch_size) + score.shape[0])] = labels

    # get the mean of the predictions for the model
    final_val_predictions = np.mean(val_predictions, axis=0)
    final_val_labels = val_labels[0, 0:len(validation_set), :] # labels are the same so just need one set

    # drop the blank rows
    if args.gpu_count > 1:
        # keras multigpu models require all batches in the prediction run to be full, so we drop the padded predictions
        if images.shape[0] < val_generator.batch_size:
            final_val_predictions = val_predictions[:, :val_predictions.shape[1] - val_generator.last_batch_padding]


    # get a range between 0 and 1 by 1000ths
    rng = np.arange(0, 1, 0.001)

    # set up an array to catch individual fscores for each class
    fscores = np.zeros((rng.shape[0], 28))

    # loop through each prediction above the threshold and calculate the fscore
    logging.info('Calculating f-scores at a range of thresholds...')
    for j,k in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(final_val_predictions[:,i]>k, dtype=np.int8)
            score = hprotein.f1_score(final_val_labels[:,i], p, average='binary')
            fscores[j,i] = score

    # Make a matrix that will hold the best threshold for each class to maximize Fscore
    max_thresholds_matrix = np.empty(28)
    for i in range(28):
        max_thresholds_matrix[i] = rng[np.where(fscores[:,i] == np.max(fscores[:,i]))[0][0]]

    macro_f1 = np.mean(np.max(fscores, axis=0))

    logging.info('Probability threshold maximizing F1-score for each class:')
    logging.info(max_thresholds_matrix)
    logging.info('Macro F1 Score -> {}'.format(macro_f1))


    #
    # Get predictions
    #

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
        model, _ = hprotein.get_best_model(args.model_folder, m[0])

        if args.gpu_count > 1:
            # compile model with desired loss function
            if m[1] == 'binary_crossentropy':
                model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
            elif m[1] == 'focal_loss':
                model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=1e-4),
                                    metrics=['accuracy', hprotein.f1])

            model.summary()

        args.model_name = m[2]
        predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids,
                                                           predict_set_lbls,
                                                           model_name=args.model_name)

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

    # get the mean of the predictions for the model
    final_predictions = np.mean(predictions, axis=0)

    # drop the blank rows
    if args.gpu_count > 1:
        # keras multigpu models require all batches in the prediction run to be full, so we drop the padded predictions
        if images.shape[0] < predict_generator.batch_size:
            final_predictions = predictions[:predictions.shape[1] - predict_generator.last_batch_padding]

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
    submit.to_csv('{}/submit_{}.csv'.format(args.submission_folder, args.ensemble_name), index=False)

    # copy the submission file to gcs
    hprotein.copy_file_to_gcs('{}/submit_{}.csv'.format(args.submission_folder, args.ensemble_name),
                              'gs://hprotein/submission/submit_{}.csv'.format(args.ensemble_name))

    logging.info('Model: {} prediction run complete!'.format(args.model_label))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
