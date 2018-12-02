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

    parser.add_argument('--run_fine_tune', dest='run_fine_tune', default='True',
                        help='Text boolean to decide wheter to run fine tuning step')

    parser.add_argument('--fine_tune_epochs', dest='fine_tune_epochs', default=3, type=int,
                        help='Number of epochs to fine tune')

    parser.add_argument('--gpu_count', dest='gpu_count', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--validation_steps', dest='validation_steps', default=10, type=int,
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

    if hprotein.text_to_bool(args.run_training):
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

    # load the data
    specimen_ids, labels = hprotein.get_data(args.train_folder, args.label_folder, mode='train',
                                             filter_ids=hprotein.validation_set)
    val_specimen_ids, val_labels = hprotein.get_data(args.train_folder, args.label_folder, mode='validate',
                                                     filter_ids=hprotein.validation_set)

    # create data generators
    logging.info('Creating Hprotein training data generator...')
    training_generator = hprotein.HproteinDataGenerator(args,
                                                        args.train_folder,
                                                        specimen_ids,
                                                        labels,
                                                        shuffle=True,
                                                        augment=True)

    logging.info('Creating Hprotein validation data generator...')
    val_generator = hprotein.HproteinDataGenerator(args,
                                                   args.train_folder,
                                                   val_specimen_ids,
                                                   val_labels)

    # create checkpoint
    checkpoint = ModelCheckpoint('models/{}.model'.format(args.model_name),
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    # create the model
    model_name = 'gap_net_bn_relu'
    model = hprotein.create_model(hprotein.SHAPE, model_name=model_name)
    if args.gpu_count > 1:
        model = multi_gpu_model(model, gpus=args.gpu_count)
        use_multiprocessing = True
        workers = args.gpu_count * 2
    elif args.gpu_count == 1:
        use_multiprocessing = True
        workers = args.gpu_count * 4
    else:
        use_multiprocessing = False
        workers = 1

    # primary training run
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc', hprotein.f1])
    model.summary()

    logging.info("Start of primary training...")
    hist = model.fit_generator(training_generator,
                               steps_per_epoch=len(training_generator),
                               validation_data=val_generator,
                               validation_steps=len(val_generator),
                               epochs=args.epochs,
                               use_multiprocessing=use_multiprocessing,
                               max_queue_size=128,
                               workers=workers,
                               verbose=1,
                               callbacks=[checkpoint])

    logging.info("Start of fine tune training...")
    if hprotein.text_to_bool(args.run_fine_tune):

        if model_name is 'gap_net_bn_relu':

            checkpoint = ModelCheckpoint('models/{}_fine_tune.model'.format(args.model_name),
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
            model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
            fths = model.fit_generator(training_generator,
                                       steps_per_epoch=len(training_generator),
                                       validation_data=val_generator,
                                       validation_steps=len(val_generator),
                                       epochs=args.epochs,
                                       use_multiprocessing=use_multiprocessing,
                                       max_queue_size=128,
                                       workers=workers,
                                       verbose=1,
                                       callbacks=[checkpoint])

        else:
            logging.warning('Fine tuning not supported for model name: {}'.format(model_name))


# ---------------------------------
# runs the eval op
# ---------------------------------
def run_eval(args):

    # Log start of eval process
    logging.info('Starting run_eval...')

    # load the data
    val_specimen_ids, val_labels = hprotein.get_data(args.train_folder, args.label_folder, mode='validate',
                                                     filter_ids=hprotein.validation_set)

    # create data generator
    logging.info('Creating Hprotein validation data generator...')
    val_generator = hprotein.HproteinDataGenerator(args,
                                                   args.train_folder,
                                                   val_specimen_ids,
                                                   val_labels)

    # eval primary model
    logging.info('Loading primary training model...')
    model = load_model('models/{}.model'.format(args.model_name), custom_objects={'f1': hprotein.f1})
    model1_max_fscore_thresholds, model1_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator, save_eval=False)

    # eval fine tune model
    logging.info('Loading fine tuned training model...')
    model = load_model('models/{}_fine_tune.model'.format(args.model_name),
                       custom_objects={'f1_loss': hprotein.f1_loss, 'f1': hprotein.f1})
    model2_max_fscore_thresholds, model2_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator, save_eval=False)

    if model1_macro_f1 > model2_macro_f1:
        logging.info('Primary model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model1_max_fscore_thresholds
        np.save('models/{}_thresh.npy'.format(args.model_name), max_fscore_thresholds)
    else:
        logging.info('Fine-tune model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model2_max_fscore_thresholds
        np.save('models/{}_fine_tune_thresh.npy'.format(args.model_name), max_fscore_thresholds)

    logging.info("Finished evaluation...")


# ---------------------------------
# runs the predict op
# ---------------------------------
def run_predict(args):

    # Log start of predict process
    logging.info('Starting run_predict...')

    # Get thresholds
    logging.info('Getting correct model and thresholds...')
    if os.path.isfile('models/{}_thresh.npy'.format(args.model_name)):

        # load model
        logging.info('Loading model {}...'.format(args.model_name))
        final_model = load_model('models/{}.model'.format(args.model_name), custom_objects={'f1': hprotein.f1})

        # load thresholds
        max_thresholds_matrix = np.load('models/{}_thresh.npy'.format(args.model_name))

    elif os.path.isfile('models/{}_fine_tune_thresh.npy'.format(args.model_name)):

        # load model
        logging.info('Loading model {}_fine_tune...'.format(args.model_name))
        final_model = load_model('models/{}_fine_tune.model'.format(args.model_name), custom_objects={'f1_loss': hprotein.f1_loss})

        # load thresholds
        max_thresholds_matrix = np.load('models/{}_fine_tune_thresh.npy'.format(args.model_name))

    # get predict data
    logging.info('Reading predict test set from {}...'.format(args.predict_folder))
    predict_set_sids, predict_set_lbls = hprotein.get_predict_data(args.predict_folder, args.submission_folder)
    args.batch_size = 8
    predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids, predict_set_lbls)

    # generate predictions
    logging.info('Starting prediction run...')

    for i in range(len(predict_generator)):
        logging.info('Getting prediction batch #{}'.format(i+1))
        images, labels = predict_generator[i]

        # if the last batch is not full append blank rows
        if images.shape[0] < predict_generator.batch_size:
            blank_rows = np.zeros((predict_generator.last_batch_padding,
                                   predict_generator.shape[0],
                                   predict_generator.shape[1],
                                   predict_generator.shape[2]))
            images = np.append(images, blank_rows, axis=0)

        score = final_model.predict(images, batch_size=predict_generator.batch_size)
        predictions = np.zeros((predict_set_sids.shape[0] + predict_generator.last_batch_padding, 28))
        predictions[i * predict_generator.batch_size : i * predict_generator.batch_size + images.shape[0]] = score

    # write out the submission csv
    hprotein.write_submission_csv(args, predict_set_sids, predictions, predict_generator.last_batch_padding, max_thresholds_matrix)

    logging.info('Model: {} prediction run complete!'.format(args.model_name))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
