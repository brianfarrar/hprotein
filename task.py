# import libraries
import logging
import argparse
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
import hprotein

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
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

    parser.add_argument('--val_steps', dest='val_steps', default=8, type=int,
                        help='Number of validation steps to run ')

    parser.add_argument('--initial_lr', dest='initial_lr', default=1e-3, type=float,
                        help='Initial learning rate')

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
                        help='Number of gpus to use for training')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--val_csv', dest='val_csv', default='val_set.csv',
                        help='CSV with the validation set info.')

    parser.add_argument('--run_predict', dest='run_predict', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--use_adaptive_thresh', dest='use_adaptive_thresh', default='True',
                        help='Text boolean to decide whether to use adaptive thresholds')

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

    parser.add_argument('--log_folder', dest='log_folder', default='logs',
                        help='Folder to save tensorboard logs to')

    parser.add_argument('--copy_to_gcs', dest='copy_to_gcs', default='True',
                        help='Text boolean to decide whether to copy to gcs')

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
    logging.info('Configuring training and fine tuning options...')

    # Get validation and training image generators
    validation_set, val_generator = hprotein.get_val_generator(args)
    training_generator = hprotein.get_train_generator(args, validation_set)

    # define check point call back
    checkpoint = ModelCheckpoint('{}/{}.model'.format(args.model_folder, args.model_label),
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    # early stopping call back
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # tensorboard call back
    tb = TensorBoard(log_dir='{}/{}'.format(args.log_folder, args.model_label),
                     batch_size=args.batch_size, write_graph=False, update_freq='epoch')

    # learning rate decay call back
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=3, min_lr=1e-6, verbose=1)

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

    # if validation steps is not chosen at launch, then set it to the length of the validation data set
    if args.val_steps == -1:
        args.val_steps = len(val_generator)

    # get class weights to deal with data imbalance
    if hprotein.text_to_bool(args.use_class_weights):
        if 'over_sample' in args.label_list:
            _, cw = hprotein.create_class_weight(hprotein.over_sample_labels_dict)
        else:
            _, cw = hprotein.create_class_weight(hprotein.labels_dict)
    else:
        cw = None

    #
    # Start of training cycle
    #

    if hprotein.text_to_bool(args.run_training):

        logging.info("Start of primary training...")

        # if this is a new model, create it and compile
        if hprotein.text_to_bool(args.new_model):
            base_model = hprotein.create_model(model_name=args.model_name)

            if args.gpu_count > 1:
                model = multi_gpu_model(base_model, gpus=args.gpu_count)
            else:
                model = base_model

            # for pretrained models run a warm start first
            if args.model_name in ['ResNet50','InceptionV2Resnet','InceptionV3',
                                   'ResNet50_Large','InceptionV2Resnet_Large','InceptionV3_Large']:


                logging.info('Running a warm start...')

                # unfreeze correct layers
                first_layer, last_layer = hprotein.get_layers_to_unfreeze(args.model_name)
                hprotein.freeze_layers(base_model, first_layer, last_layer)

                # compile model with desired loss function
                if args.loss_function == 'f1_loss':
                    model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=args.initial_lr),
                                  metrics=['accuracy', hprotein.f1])
                elif args.loss_function == 'binary_crossentropy':
                    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.initial_lr),
                                  metrics=['accuracy', hprotein.f1])
                elif args.loss_function == 'focal_loss':
                    model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=args.initial_lr),
                                  metrics=['accuracy', hprotein.f1])

                model.summary()

                # warms start constants
                warm_start_epochs = 4
                warm_start_lr_change_point = 2

                # define learning rate schedule callback for warm start
                lr_schedule = hprotein.lr_decay_schedule(initial_lr=args.initial_lr,
                                                         change_point=warm_start_lr_change_point)
                schedule = LearningRateScheduler(schedule=lr_schedule)

                # run a fit_generator step
                hist = model.fit_generator(training_generator,
                                           steps_per_epoch=warm_start_epochs,
                                           validation_data=val_generator,
                                           validation_steps=8,
                                           epochs=warm_start_epochs,
                                           use_multiprocessing=use_multiprocessing,
                                           workers=workers,
                                           class_weight=cw,
                                           verbose=1,
                                           callbacks=[checkpoint, schedule, early_stop])

                # unfreeze all layers
                if args.gpu_count > 1:
                    for layer in base_model.layers:
                        layer.trainable = True
                else:
                    for layer in model.layers:
                        layer.trainable = True

            # compile model with desired loss function
            if args.loss_function == 'f1_loss':
                model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=args.initial_lr),
                              metrics=['accuracy', hprotein.f1])
            elif args.loss_function == 'binary_crossentropy':
                model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.initial_lr),
                              metrics=['accuracy', hprotein.f1])
            elif args.loss_function == 'focal_loss':
                model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=args.initial_lr),
                              metrics=['accuracy', hprotein.f1])

        # if this is a run on an existing model, then load and compile
        logging.info('Starting full model training...')
        if not hprotein.text_to_bool(args.new_model):
            model, base_model = hprotein.prepare_existing_model(args, lr=args.initial_lr)

        model.summary()

        # define learning rate schedule callback
        lr_schedule = hprotein.lr_decay_schedule(initial_lr=args.initial_lr, change_point=args.change_lr_epoch)
        schedule = LearningRateScheduler(schedule=lr_schedule)

        hist = model.fit_generator(training_generator,
                                   steps_per_epoch=args.steps_per_epoch,
                                   validation_data=val_generator,
                                   validation_steps=8,
                                   epochs=args.epochs,
                                   use_multiprocessing=use_multiprocessing,
                                   workers=workers,
                                   class_weight=cw,
                                   verbose=1,
                                   callbacks=[checkpoint, lr_decay, early_stop, tb])

        if args.gpu_count > 1:
            base_model.save('{}/{}.model'.format(args.model_folder, args.model_label))

    #
    # Start of fine tune cycle
    #

    if hprotein.text_to_bool(args.run_fine_tune):

        logging.info("Start of fine tune training...")

        # set up checkpoint call back
        checkpoint = ModelCheckpoint('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                                     monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min', period=1)

        # if we did not run training and are only fine tuning, load model
        if not hprotein.text_to_bool(args.run_training):
            model, base_model = hprotein.prepare_existing_model(args, lr=args.initial_lr/10.)

        # freeze correct layers
        first_layer, last_layer = hprotein.get_layers_to_unfreeze(args.model_name)
        hprotein.freeze_layers(base_model, first_layer, last_layer)

        # compile model with desired loss function
        if args.loss_function == 'f1_loss':
            model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=args.initial_lr),
                          metrics=['accuracy', hprotein.f1])
        elif args.loss_function == 'binary_crossentropy':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.initial_lr),
                          metrics=['accuracy', hprotein.f1])
        elif args.loss_function == 'focal_loss':
            model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=args.initial_lr),
                          metrics=['accuracy', hprotein.f1])

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

        if args.gpu_count > 1:
            base_model.save('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label))

        # copy models to gcs
        if hprotein.text_to_bool(args.copy_to_gcs):

            hprotein.copy_file_to_gcs('{}/{}.model'.format(args.model_folder, args.model_label),
                                      'gs://hprotein/models/{}.model'.format(args.model_label))

            hprotein.copy_file_to_gcs('{}/{}_fine_tune.model'.format(args.model_folder, args.model_label),
                                      'gs://hprotein/models/{}_fine_tune.model'.format(args.model_label))



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
    model, base_model = hprotein.prepare_existing_model(args, lr=args.initial_lr)
    model.summary()

    model1_max_fscore_thresholds, model1_macro_f1 = hprotein.get_max_fscore_matrix(args, model, val_generator, save_eval=True)

    # eval fine tune model
    logging.info('Loading fine tuned training model...')
    model, base_model = hprotein.prepare_existing_model(args, lr=args.initial_lr, fine_tune=True)
    model.summary()

    model2_max_fscore_thresholds, model2_macro_f1 = hprotein.get_max_fscore_matrix(args, model, val_generator, save_eval=True)

    if model1_macro_f1 > model2_macro_f1:
        logging.info('Primary model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model1_max_fscore_thresholds
        np.save('{}/{}_thresh.npy'.format(args.model_folder, args.model_label), max_fscore_thresholds)
        if hprotein.text_to_bool(args.copy_to_gcs):
            hprotein.copy_file_to_gcs('{}/{}_thresh.npy'.format(args.model_folder, args.model_label),
                                      'gs://hprotein/models/{}_thresh.npy'.format(args.model_label))
    else:
        logging.info('Fine-tune model has a better Macro-F1, saving thresholds...')
        max_fscore_thresholds = model2_max_fscore_thresholds
        np.save('{}/{}_fine_tune_thresh.npy'.format(args.model_folder, args.model_label), max_fscore_thresholds)
        if hprotein.text_to_bool(args.copy_to_gcs):
            hprotein.copy_file_to_gcs('{}/{}_fine_tune_thresh.npy'.format(args.model_folder, args.model_label),
                                      'gs://hprotein/models/{}_fine_tune_thresh.npy'.format(args.model_label))

    logging.info("Finished evaluation...")


# ---------------------------------
# runs the predict op
# ---------------------------------
def run_predict(args):

    # Log start of predict process
    logging.info('Starting run_predict...')

    # get predict data
    logging.info('Reading predict test set from {}...'.format(args.predict_folder))
    predict_set_sids, predict_set_lbls = hprotein.get_data(args.submission_folder, args.submission_list, mode='test')
    predict_generator = hprotein.HproteinDataGenerator(args, args.predict_folder, predict_set_sids, predict_set_lbls,
                                                       model_name=args.model_name, mode='predict')

    # get the best model and related threshold matrix
    final_model, max_thresholds_matrix = hprotein.get_best_model(args.model_folder, args.model_label)

    if args.gpu_count > 1:
        # compile model with desired loss function
        if args.loss_function == 'f1_loss':
            final_model.compile(loss=hprotein.f1_loss, optimizer=Adam(lr=args.initial_lr),
                                metrics=['accuracy', hprotein.f1])
        elif args.loss_function == 'binary_crossentropy':
            final_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.initial_lr),
                                metrics=['accuracy', hprotein.f1])
        elif args.loss_function == 'focal_loss':
            final_model.compile(loss=hprotein.focal_loss, optimizer=Adam(lr=args.initial_lr),
                                metrics=['accuracy', hprotein.f1])

        final_model.summary()

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

    hprotein.write_submission_csv(args, submit, predictions, max_thresholds_matrix)

    logging.info('Model: {} prediction run complete!'.format(args.model_label))


# ---------------------------------
# main runner
# ---------------------------------
def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
