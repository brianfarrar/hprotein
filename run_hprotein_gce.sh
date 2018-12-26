#!/usr/bin/env bash
python3 task.py \
    --run_training="True" \
    --batch_size=64 \
    --epochs=30 \
    --steps_per_epoch=-1 \
    --initial_lr=1e-3 \
    --change_lr_epoch=20 \
    --use_class_weights="True" \
    --run_fine_tune="True" \
    --fine_tune_epochs=2 \
    --loss_function="focal_loss" \
    --gpu_count=2 \
    --run_eval="True" \
    --val_csv="val_set_kfold_4.csv" \
    --run_predict="True" \
    --new_model="True" \
    --model_name="gap_net_bn_relu" \
    --model_label="model_kf4" \
    --model_folder="models" \
    --train_folder="stage1_train" \
    --label_folder="stage1_labels" \
    --label_list="train.csv" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --submission_list="sample_submission.csv" \
    --copy_to_gcs="True"
