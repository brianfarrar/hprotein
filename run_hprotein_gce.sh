#!/usr/bin/env bash
python3 task.py \
    --run_training="True" \
    --batch_size=64 \
    --epochs=1 \
    --steps_per_epoch=-1 \
    --change_lr_epoch=1 \
    --run_fine_tune="True" \
    --fine_tune_epochs=2 \
    --loss_function="focal_loss" \
    --gpu_count=2 \
    --run_eval="True" \
    --val_set="val_set_combo.csv" \
    --run_predict="True" \
    --new_model="False" \
    --model_name="gap_net_bn_relu" \
    --model_label="model_573537" \
    --model_folder="models" \
    --train_folder="stage1_train_combo" \
    --label_folder="stage1_labels" \
    --label_list="train_combo.csv" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --submission_list="sample_submission.csv" \
    --job_id=''
