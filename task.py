# import libraries
import logging
import argparse
import json
import os
import uuid
import numpy as np
import hprotein
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

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

    parser.add_argument('--output_folder', dest='output_folder', default='./output',
                        help='Folder to output images to')

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
        args.model_folder = '{}_{}'.format(args.model_folder, unique_id)
        args.output_folder = '{}_{}'.format(args.output_folder, unique_id)
    else:
        unique_id = args.model_folder.split('_')[-1]
        args.output_folder = '{}_{}'.format(args.output_folder, unique_id)

    if hprotein.text_to_bool(args.run_training):
        run_training(args, unique_id)

    if hprotein.text_to_bool(args.run_eval):
        run_eval(args, unique_id)

    if hprotein.text_to_bool(args.run_predict):
        run_predict(args, unique_id)


# ---------------------------------
# runs the train op
# ---------------------------------
def run_training(args, unique_id):

    # Log start of training process
    logging.info('Starting run_training...')

    # get data
    specimen_ids, labels = hprotein.get_train_data(args.train_folder, args.label_folder)

    # get training features and labels
    #TODO: set up train/validation split regimen
    train_set_sids = specimen_ids[0:3]
    train_set_lbls = labels[0:3]

    # get validation features and labels
    val_set_sids = specimen_ids[3:]
    val_set_lbls = labels[3:]

    # create data generators
    training_generator = hprotein.HproteinDataGenerator(args, train_set_sids, train_set_lbls)
    val_generator = hprotein.HproteinDataGenerator(args, val_set_sids, val_set_lbls)

    # create checkpoint and learning rate modifier
    checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

    # create the model
    model = hprotein.create_model(hprotein.SHAPE)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc', hprotein.f1])
    model.summary()

    hist = model.fit_generator(training_generator,
                               steps_per_epoch=len(training_generator),
                               validation_data=val_generator,
                               validation_steps=args.validation_steps,
                               epochs=args.epochs,
                               use_multiprocessing=False,
                               workers=1,
                               verbose=1,
                               callbacks=[checkpoint])

# ---------------------------------
# runs the eval op
# ---------------------------------
def run_eval(args, unique_id):

    # Log start of training process
    logging.info('Starting run_eval...')


# ---------------------------------
# runs the predict op
# ---------------------------------
def run_predict(args, unique_id):

    # Log start of training process
    logging.info('Starting run_predict...')


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
