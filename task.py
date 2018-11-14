# import libraries
import logging
import argparse
import json
import os
import uuid
import numpy as np
import hprotein

# -------------------------------------------------------------
# Runs the model.  This is the main runner for this package.
# -------------------------------------------------------------
def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_training', dest='run_training', default='True',
                        help='Text boolean to decide whether to run training')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--run_ensemble_eval', dest='run_ensemble_eval', default='True',
                        help='Text boolean to decide whether to run ensemble eval')

    parser.add_argument('--run_predict', dest='run_predict', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--new_model', dest='new_model', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--model_name', dest='model_name', default='base',
                        help='Model to run.')

    parser.add_argument('--model_folder', dest='model_folder', default='./',
                        help='Folder to store the model checkpoints')

    parser.add_argument('--output_folder', dest='output_folder', default='./output',
                        help='Folder to output images to')

    parser.add_argument('--submission_folder', dest='submission_folder', default='./submission',
                        help='Folder for submission csv')

    parser.add_argument('--zip_folder', dest='zip_folder', default='./zips',
                        help='Folder that contains the zipped data files')

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
