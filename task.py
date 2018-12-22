# import libraries
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

    parser.add_argument('--run_training', dest='run_training', default='True',
                        help='Text boolean to decide whether to run training')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='Number of examples per batch')

    parser.add_argument('--epochs', dest='epochs', default=10, type=int,
                        help='Number of epochs')

    parser.add_argument('--steps_per_epoch', dest='steps_per_epoch', default=-1, type=int,
                        help='Number of epochs')

    parser.add_argument('--change_lr_epoch', dest='change_lr_epoch', default=16, type=int,
                        help='Epoch to reduce learning rate')

    parser.add_argument('--use_class_weights', dest='use_class_weights', default='False',
                        help='Text boolean to decide whether to use class weights in training')

    parser.add_argument('--run_fine_tune', dest='run_fine_tune', default='False',
                        help='Text boolean to decide wheter to run fine tuning step')

    parser.add_argument('--fine_tune_epochs', dest='fine_tune_epochs', default=3, type=int,
                        help='Number of epochs to fine tune')

    parser.add_argument('--loss_function', dest='loss_function', default='focal_loss',
                        help='Which loss function to use for the run')

    parser.add_argument('--gpu_count', dest='gpu_count', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--val_csv', dest='val_csv', default='val_set.csv',
                        help='Model to run.')

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

    parser.add_argument('--job_id', dest='job_id', default='',
                        help='Unique job id')

    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # If we are starting a new model, then get a new unique id
    if hprotein.text_to_bool(args.new_model):
        unique_id = str(uuid.uuid4())[-6:]
        logging.info('New model number -> {}'.format(unique_id))
        args.model_label = '{}_{}'.format(args.model_label, unique_id)
    else:
        logging.info('Continuing training for model: {}'.format(args.model_label))

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

    # Get validation and training image generators
    validation_set, val_generator = hprotein.get_val_generator(args)
    training_generator = hprotein.get_train_generator(args, validation_set)

    # define check point call back
    checkpoint = ModelCheckpoint('{}/{}.model'.format(args.model_folder, args.model_label),
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    # define learning rate schedule callback
    lr_schedule = hprotein.lr_decay_schedule(change_point=args.change_lr_epoch)
    schedule = LearningRateScheduler(schedule=lr_schedule)

    # configure trainer options for the environment
    if args.gpu_count > 1:
        use_multiprocessing = True
        workers = args.gpu_count * 2
    elif args.gpu_count == 1:
        use_multiprocessing = True
        workers = args.gpu_count * 4
    else:
        use_multiprocessing = False
        workers = 1

    # if steps per epoch is not chosen at launch, then set it to the length of the full data set
    if args.steps_per_epoch == -1:
        args.steps_per_epoch = len(training_generator)

    # get class weights to deal with data imbalance
    if hprotein.text_to_bool(args.use_class_weights):
        _, cw = hprotein.create_class_weight(hprotein.labels_dict)
    else:
        cw = None

    # create the base model
    base_model = hprotein.create_model(model_name=args.model_name)

    # create or load the model
    if hprotein.text_to_bool(args.new_model):
        if args.gpu_count > 1:
            model = multi_gpu_model(base_model, gpus=args.gpu_count)
        else:
            model = base_model
    else:
        if args.loss_function == 'binary_crossentropy':
            model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                               custom_objects={'f1': hprotein.f1})
        elif args.loss_function == 'focal_loss':
            model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                               custom_objects={'f1': hprotein.f1, 'focal_loss': hprotein.focal_loss})

    # compile model with desired loss function
    if args.loss_function == 'focal_loss':
        loss = hprotein.focal_loss
    elif args.loss_function == 'binary_crossentropy':
        loss = args.loss_function

    if hprotein.text_to_bool(args.run_training):

        # primary training run
        model.compile(loss=loss, optimizer=Adam(lr=1e-03), metrics=['acc', hprotein.f1])
        model.summary()

        logging.info("Start of primary training...")
        hist = model.fit_generator(training_generator,
                                   steps_per_epoch=args.steps_per_epoch,
                                   validation_data=val_generator,
                                   validation_steps=8,
                                   epochs=args.epochs,
                                   use_multiprocessing=use_multiprocessing,
                                   workers=workers,
                                   class_weight=cw,
                                   verbose=1,
                                   callbacks=[checkpoint, schedule])

    logging.info("Start of fine tune training...")
    if hprotein.text_to_bool(args.run_fine_tune):

        # if we did not run training and are only fine tuning, load model
        if not hprotein.text_to_bool(args.run_training):
            if args.loss_function == 'binary_crossentropy':
                model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                                   custom_objects={'f1': hprotein.f1})
            elif args.loss_function == 'focal_loss':
                model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                                   custom_objects={'f1': hprotein.f1, 'focal_loss': hprotein.focal_loss})

        if args.model_name == 'gap_net_bn_relu':

            checkpoint = ModelCheckpoint('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                                         monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='min', period=1)

            if args.gpu_count > 1:
                for layer in base_model.layers:
                    layer.trainable = False

                base_model.layers[-1].trainable = True
                base_model.layers[-2].trainable = True
                base_model.layers[-3].trainable = True
                base_model.layers[-4].trainable = True
                base_model.layers[-5].trainable = True
                base_model.layers[-6].trainable = True
                base_model.layers[-7].trainable = True
            else:
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

            checkpoint = ModelCheckpoint('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                                         monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='min', period=1)

            if args.gpu_count > 1:
                for layer in base_model.layers:
                    layer.trainable = False

                base_model.layers[-3].trainable = True
                base_model.layers[-4].trainable = True
                base_model.layers[-5].trainable = True
                base_model.layers[-6].trainable = True
                base_model.layers[-7].trainable = True
                base_model.layers[-8].trainable = True
                base_model.layers[-9].trainable = True
            else:
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

        # compile model with desired loss function
        if args.loss_function == 'binary_crossentropy':
            model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])
        elif args.loss_function == 'focal_loss':
            model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', hprotein.f1])

        model.summary()

        fths = model.fit_generator(training_generator,
                                   steps_per_epoch=args.steps_per_epoch,
                                   validation_data=val_generator,
                                   validation_steps=8,
                                   epochs=args.fine_tune_epochs,
                                   use_multiprocessing=use_multiprocessing,
                                   max_queue_size=4,
                                   workers=workers,
                                   class_weight=cw,
                                   verbose=1,
                                   callbacks=[checkpoint])



# ---------------------------------
# runs the eval op
# ---------------------------------
def run_eval(args):

    # Log start of eval process
    logging.info('Starting run_eval...')

    # create data generator
    logging.info('Creating Hprotein validation data generator...')
    # Get validation and training image generators
    _, val_generator = hprotein.get_val_generator(args)

    # eval primary model
    logging.info('Loading primary training model...')
    if args.loss_function == 'binary_crossentropy':
        model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                           custom_objects={'f1': hprotein.f1})
    elif args.loss_function == 'focal_loss':
        model = load_model('{}/{}.model'.format(args.model_folder, args.model_label),
                           custom_objects={'f1': hprotein.f1, 'focal_loss': hprotein.focal_loss})

    model1_max_fscore_thresholds, model1_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator)

    # eval fine tune model
    logging.info('Loading fine tuned training model...')

    if args.loss_function == 'binary_crossentropy':
        model = load_model('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                           custom_objects={'f1_loss': hprotein.f1_loss, 'f1': hprotein.f1})
    elif args.loss_function == 'focal_loss':
        model = load_model('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                           custom_objects={'f1_loss': hprotein.f1_loss, 'f1': hprotein.f1,
                                           'focal_loss': hprotein.focal_loss})

    model2_max_fscore_thresholds, model2_macro_f1 = hprotein.get_max_fscore_matrix(model, val_generator)

    if model1_macro_f1 > model2_macro_f1:
        logging.info('Primary model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model1_max_fscore_thresholds
        np.save('{}/{}_thresh.npy'.format(args.model_folder, args.model_label), max_fscore_thresholds)
    else:
        logging.info('Fine-tune model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model2_max_fscore_thresholds
        np.save('{}/{}_fine_tune_thresh.npy'.format(args.model_folder, args.model_label), max_fscore_thresholds)

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
    predict_set_sids, predict_set_lbls = hprotein.get_data(args.submission_folder, args.submission_list, mode='test')
    predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids, predict_set_lbls,
                                                       model_name=args.model_name)

    # generate predictions
    logging.info('Starting prediction run...')

    # read in the list of samples in the correct order to submit
    submit = pd.read_csv('{}/{}'.format(args.submission_folder, args.submission_list))

    # create an empty array to catch the predictions
    if args.gpu_count > 1:
        # keras multigpu models require all batches in the prediction run to be full, so we pad for the last batch
        predictions = np.zeros((predict_set_sids.shape[0] + predict_generator.last_batch_padding, 28))
    else:
        predictions = np.zeros((predict_set_sids.shape[0], 28))

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

        score = final_model.predict(images)
        predictions[i * predict_generator.batch_size : ((i * predict_generator.batch_size) + score.shape[0])] = score

    # drop the blank rows
    if args.gpu_count > 1:
        # keras multigpu models require all batches in the prediction run to be full, so we drop the padded predictions
        if images.shape[0] < predict_generator.batch_size:
            predictions = predictions[:predictions.shape[1] - predict_generator.last_batch_padding]

    # convert the predictions into the submission file format
    logging.info('Converting to submission format...')
    prediction_str = []
    for row in tqdm(range(submit.shape[0])):
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
    submit.to_csv('{}/submit_{}.csv'.format(args.submission_folder, args.model_label), index=False)

    logging.info('Model: {} prediction run complete!'.format(args.model_label))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
