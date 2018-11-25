# import libraries
import logging
import argparse
import json
import os
import uuid
import numpy as np
import pandas as pd
import hprotein
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model


# -------------------------------------------------------------
# Runs the model.  This is the main runner for this package.
# -------------------------------------------------------------
def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_training', dest='run_training', default='True',
                        help='Text boolean to decide whether to run training')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='Number of examples per batch')

    parser.add_argument('--epochs', dest='epochs', default=10, type=int,
                        help='Number of epochs')

    parser.add_argument('--gpu_count', dest='gpu_count', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--validation_steps', dest='validation_steps', default=10, type=int,
                        help='Number of validation_steps')

    parser.add_argument('--validation_split', dest='validation_split', default=3000, type=int,
                        help='Number of validation_steps')

    parser.add_argument('--run_predict', dest='run_predict', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--new_model', dest='new_model', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--model_name', dest='model_name', default='base',
                        help='Model to run.')

    parser.add_argument('--model_folder', dest='model_folder', default='./',
                        help='Folder to store the model checkpoints')

    parser.add_argument('--label_folder', dest='label_folder', default='./stage1_labels',
                        help='Folder that contains the train label csv')

    parser.add_argument('--train_folder', dest='train_folder', default='./stage1_data',
                        help='Folder that contains the train data files')

    parser.add_argument('--predict_folder', dest='predict_folder', default='./stage1_predict',
                        help='Folder that contains the data files to predict')

    parser.add_argument('--submission_folder', dest='submission_folder', default='./submission',
                        help='Folder for submission csv')

    parser.add_argument('--job_id', dest='job_id', default='',
                        help='Unique job id')

    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # If we are starting a new model, then get a new unique id otherwise, get it from the model folder name
    if hprotein.text_to_bool(args.new_model):
        unique_id = str(uuid.uuid4())[-6:]
        logging.info('New model number -> {}'.format(unique_id))
        args.model_name = '{}_{}'.format(args.model_name, unique_id)
    else:
        unique_id = args.model_name.split('_')[-1]

    # get data and create train/test split

    train_specimen_ids, \
    val_specimen_ids, \
    train_labels, \
    val_labels = hprotein.get_train_test_split(args, test_size=args.validation_split)

    # create data generators
    logging.info('Creating Hprotein training data generator...')
    training_generator = hprotein.HproteinDataGenerator(args, args.train_folder, train_specimen_ids, train_labels)
    logging.info('Creating Hprotein validation data generator...')
    val_generator = hprotein.HproteinDataGenerator(args, args.train_folder, val_specimen_ids, val_labels)

    if hprotein.text_to_bool(args.run_training):
        run_training(args, training_generator, val_generator)

    if hprotein.text_to_bool(args.run_eval):
        max_thresholds_matrix = run_eval(args, val_generator)

    if hprotein.text_to_bool(args.run_predict):
        run_predict(args, max_thresholds_matrix)


# ---------------------------------
# runs the train op
# ---------------------------------
def run_training(args, training_generator, val_generator):

    # Log start of training process
    logging.info('Starting run_training...')

    # create checkpoint and learning rate modifier
    checkpoint = ModelCheckpoint('{}.model'.format(args.model_name),
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

    # create the model
    model = hprotein.create_model(hprotein.SHAPE)
    if args.gpu_count > 1:
        model = multi_gpu_model(model, gpus=args.gpu_count)

    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc', hprotein.f1])

    model.summary()

    hist = model.fit_generator(training_generator,
                               steps_per_epoch=len(training_generator),
                               validation_data=val_generator,
                               validation_steps=args.validation_steps,
                               epochs=args.epochs,
                               use_multiprocessing=True,
                               max_queue_size=128,
                               workers=16,
                               verbose=1,
                               callbacks=[checkpoint,reduceLROnPlato])


# ---------------------------------
# runs the eval op
# ---------------------------------
def run_eval(args, val_generator):

    # Log start of eval process
    logging.info('Starting run_eval...')

    # loading model
    final_model = load_model('{}.model'.format(args.model_name), custom_objects={'f1': hprotein.f1})

    # create empty arrays to receive the predictions and  labels
    val_predictions = np.empty((0, 28))
    val_labels = np.empty((0, 28))

    # loop through the validation data and make predictions
    for i in range(len(val_generator)):
        image, label = val_generator[i]
        scores = final_model.predict(image, batch_size=8)  # batch size reduced to avoid OOM issue
        val_predictions = np.append(val_predictions, scores, axis=0)
        val_labels = np.append(val_labels, label, axis=0)

    max_fscore_thresholds = hprotein.get_max_fscore_matrix(val_predictions, val_labels)

    logging.info("Finished evaluation...")

    return max_fscore_thresholds


# ---------------------------------
# runs the predict op
# ---------------------------------
def run_predict(args, max_thresholds_matrix):

    # Log start of predict process
    logging.info('Starting run_predict...')

    # loading model
    logging.info('Loading model {}...'.format(args.model_name))
    final_model = load_model('{}.model'.format(args.model_name), custom_objects={'f1': hprotein.f1})

    # get predict data
    logging.info('Reading predict test set from {}...'.format(args.predict_folder))
    predict_set_sids, predict_set_lbls = hprotein.get_predict_data(args.predict_folder, args.submission_folder)
    logging.info('Predict rows -> {}'.format(len(predict_set_sids)))
    args.batch_size = 2
    predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids, predict_set_lbls)
    logging.info('Predict batches -> {}'.format(len(predict_generator)))

    # generate predictions
    logging.info('Starting prediction run...')
    predictions = np.zeros((predict_set_sids.shape[0], 28))
    for i in range(len(predict_generator)):
        logging.info('Getting prediction batch #{}'.format(i+1))
        images, labels = predict_generator[i]
        logging.info('Image Count -> {}'.format(len(images)))
        logging.info('Label Count -> {}'.format(len(labels)))
        score = final_model.predict(images, batch_size=args.batch_size)
        predictions[i * args.batch_size : i * args.batch_size + score.shape[0]] = score

    # get the list of submission specimen ids required
    submit_data = pd.read_csv(args.submission_folder + '/sample_submission.csv')

    # get the subset of labels that match the specimen images that are on TEST_PATH
    submit_data = submit_data.loc[submit_data['Id'].isin(predict_set_sids)]
    tmp_sid_list = submit_data['Id'].values

    prediction_str = []

    logging.info('Reformatting predictions and generating submission format...')

    for i in range(predictions.shape[0]):
        logging.info('Writing prediction #{} for specimen_id: {}'.format(i+1, tmp_sid_list[i]))
        submit_str = ' '
        for j in range(predictions.shape[1]):
            if predictions[i, j] >= max_thresholds_matrix[j]:
                submit_str += str(j) + ' '

        prediction_str.append(submit_str.strip())

    submit_data['Predicted'] = np.array(prediction_str)

    submit_data.to_csv(args.submission_folder + '/submit_{}.csv'.format(args.model_name), index=False)

    logging.info('Prediction run complete!')


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
