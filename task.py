# import libraries
import logging
import argparse
import os
import uuid
import numpy as np
import pandas as pd
import hprotein

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model


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

    parser.add_argument('--run_fine_tune', dest='run_fine_tune', default='True',
                        help='Text boolean to decide wheter to run fine tuning step')

    parser.add_argument('--fine_tune_epochs', dest='fine_tune_epochs', default=3, type=int,
                        help='Number of epochs to fine tune')

    parser.add_argument('--gpu_count', dest='gpu_count', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--val_csv', dest='val_csv', default='val_set.csv',
                        help='Model to run.')

    parser.add_argument('--validation_steps', dest='validation_steps', default=10, type=int,
                        help='Number of validation_steps')

    parser.add_argument('--run_predict', dest='run_predict', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--new_model', dest='new_model', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--model_name', dest='model_name', default='gap_net_bn_relu',
                        help='Network to run.')

    parser.add_argument('--model_label', dest='model_label', default='base',
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

    # If we are starting a new model, then get a new unique id
    if hprotein.text_to_bool(args.new_model):
        unique_id = str(uuid.uuid4())[-6:]
        logging.info('New model number -> {}'.format(unique_id))
        args.model_label = '{}_{}'.format(args.model_label, unique_id)

    if hprotein.text_to_bool(args.run_training) or hprotein.text_to_bool(args.run_fine_tune):
        run_training(args)

    if hprotein.text_to_bool(args.run_eval):
        run_eval(args)

    if hprotein.text_to_bool(args.run_predict):
        run_predict(args)


# ---------------------------------
# runs the train op
# ---------------------------------
def run_training(args):

    # Log start of training process
    logging.info('Starting run_training...')

    # get the validation set
    df_valid = pd.read_csv(args.val_csv)
    validation_set = df_valid.values.tolist()
    validation_set = [item for sublist in validation_set for item in sublist]

    # load the data
    specimen_ids, labels = hprotein.get_data(args.train_folder, args.label_folder, mode='train',
                                             filter_ids=validation_set)
    val_specimen_ids, val_labels = hprotein.get_data(args.train_folder, args.label_folder, mode='validate',
                                                     filter_ids=validation_set)

    # create data generators
    logging.info('Creating Hprotein training data generator...')
    training_generator = hprotein.HproteinDataGenerator(args,
                                                        args.train_folder,
                                                        specimen_ids,
                                                        labels,
                                                        model_name=args.model_name,
                                                        shuffle=True,
                                                        augment=True)

    logging.info('Creating Hprotein validation data generator...')
    val_generator = hprotein.HproteinDataGenerator(args,
                                                   args.train_folder,
                                                   val_specimen_ids,
                                                   val_labels,
                                                   model_name=args.model_name)

    # create checkpoint
    checkpoint = ModelCheckpoint('models/{}.model'.format(args.model_label),
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    # create the model
    model = hprotein.create_model(model_name=args.model_name)
    use_multiprocessing = False
    workers = 1

    if hprotein.text_to_bool(args.run_training):
        # primary training run
        model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc', hprotein.f1])
        model.summary()

        logging.info("Start of primary training...")
        hist = model.fit_generator(training_generator,
                                   steps_per_epoch=len(training_generator),
                                   validation_data=val_generator,
                                   validation_steps=8,
                                   epochs=args.epochs,
                                   use_multiprocessing=use_multiprocessing,
                                   workers=workers,
                                   verbose=1,
                                   callbacks=[checkpoint])

    logging.info("Start of fine tune training...")
    if hprotein.text_to_bool(args.run_fine_tune):

        if not hprotein.text_to_bool(args.run_training):
            model = load_model('models/{}.model'.format(args.model_label), custom_objects={'f1': hprotein.f1})

        if args.model_name == 'gap_net_bn_relu':

            checkpoint = ModelCheckpoint('models/{}_fine_tune.model'.format(args.model_label),
                                         monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='min', period=1)

            for layer in model.layers:
                layer.trainable = False

            model.layers[-1].trainable = True
            model.layers[-2].trainable = True
            model.layers[-3].trainable = True
            model.layers[-4].trainable = True
            model.layers[-5].trainable = True
            model.layers[-6].trainable = True
            model.layers[-7].trainable = True

        elif args.model_name == 'InceptionV2Resnet':

            checkpoint = ModelCheckpoint('models/{}_fine_tune.model'.format(args.model_label),
                                         monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='min', period=1)

            for layer in model.layers:
                layer.trainable = False

            model.layers[-3].trainable = True
            model.layers[-4].trainable = True
            model.layers[-5].trainable = True
            model.layers[-6].trainable = True
            model.layers[-7].trainable = True
            model.layers[-8].trainable = True
            model.layers[-9].trainable = True


        else:
            logging.warning('Fine tuning not supported for model name: {}'.format(args.model_name))

        model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
        model.summary()
        fths = model.fit_generator(training_generator,
                                   steps_per_epoch=len(training_generator),
                                   validation_data=val_generator,
                                   validation_steps=8,
                                   epochs=args.fine_tune_epochs,
                                   use_multiprocessing=use_multiprocessing,
                                   max_queue_size=4,
                                   workers=workers,
                                   verbose=1,
                                   callbacks=[checkpoint])


# ---------------------------------
# runs the eval op
# ---------------------------------
def run_eval(args):

    # Log start of eval process
    logging.info('Starting run_eval...')

    # get the validation set
    df_valid = pd.read_csv(args.val_csv)
    validation_set = df_valid.values.tolist()
    validation_set = [item for sublist in validation_set for item in sublist]

    # load the data
    val_specimen_ids, val_labels = hprotein.get_data(args.train_folder, args.label_folder, mode='validate',
                                                     filter_ids=validation_set)

    # create data generator
    logging.info('Creating Hprotein validation data generator...')
    val_generator = hprotein.HproteinDataGenerator(args,
                                                   args.train_folder,
                                                   val_specimen_ids,
                                                   val_labels,
                                                   model_name=args.model_name)

    # eval primary model
    logging.info('Loading primary training model...')
    model = load_model('models/{}.model'.format(args.model_label), custom_objects={'f1': hprotein.f1})
    model1_max_fscore_thresholds, model1_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator, save_eval=False)

    # eval fine tune model
    logging.info('Loading fine tuned training model...')
    model = load_model('models/{}_fine_tune.model'.format(args.model_label),
                       custom_objects={'f1_loss': hprotein.f1_loss, 'f1': hprotein.f1})
    model2_max_fscore_thresholds, model2_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator, save_eval=False)

    if model1_macro_f1 > model2_macro_f1:
        logging.info('Primary model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model1_max_fscore_thresholds
        np.save('models/{}_thresh.npy'.format(args.model_label), max_fscore_thresholds)
    else:
        logging.info('Fine-tune model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model2_max_fscore_thresholds
        np.save('models/{}_fine_tune_thresh.npy'.format(args.model_label), max_fscore_thresholds)

    logging.info("Finished evaluation...")


# ---------------------------------
# runs the predict op
# ---------------------------------
def run_predict(args):

    # Log start of predict process
    logging.info('Starting run_predict...')

    # get the best model and related threshold matrix
    final_model, max_thresholds_matrix = hprotein.get_best_model(args)

    # get predict data
    logging.info('Reading predict test set from {}...'.format(args.predict_folder))
    predict_set_sids, predict_set_lbls = hprotein.get_data(args.predict_folder, args.submission_folder, mode='test',
                                                           filter_ids=None)
    predict_generator = hprotein.HproteinDataGenerator(args,
                                                       args.predict_folder,
                                                       predict_set_sids,
                                                       predict_set_lbls,
                                                       model_name=args.model_name)

    # generate predictions
    logging.info('Starting prediction run...')

    # read in the list of samples in the correct order to submit
    submit = pd.read_csv('{}/sample_submission.csv'.format(args.submission_folder))

    # create an empty array to catch the predictions
    predictions = np.zeros(predict_set_sids.shape[0], 28)

    # get the predictions
    for i in range(len(predict_generator)):
        if i % 10 == 0:
            logging.info('Predicting batch {} of {}'.format(i, len(predict_generator)))
        images, labels = predict_generator[i]
        score = final_model.predict(images)
        predictions[i * predict_generator.batch_size : ((i * predict_generator.batch_size) + score.shape[0])] = score

    # convert the predictions into the submission file format
    prediction_str = []
    for row in range(submit.shape[0]):
        if row % 10 == 0:
            logging.info('Converting labels for prediction {} of {}'.format(row, submit.shape[0]))
        str_label = ''
        for col in range(predictions.shape[1]):
            if predictions[row, col] < max_thresholds_matrix[col]:
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction_str.append(str_label.strip())

    # add column to pandas dataframe for submission
    submit['Predicted'] = np.array(prediction_str)

    # write out the csv
    submit.to_csv('{}/{}.csv.csv'.format(args.submission_folder, args.model_label), index=False)

    logging.info('Model: {} prediction run complete!'.format(args.model_label))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
