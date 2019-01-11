#!/usr/bin/env bash
python3 ensemble.py \
    --ensemble_name="ensemble" \
    --ensemble_csv="" \
    --gen_predictions="False" \
    --batch_size=32 \
    --use_class_weights="True" \
    --gpu_count=2 \
    --thresh=-1 \
    --val_csv="val_set_kfold_4.csv" \
    --train_folder="stage1_train" \
    --model_name="gap_net_bn_relu" \
    --model_label="model" \
    --model_folder="models" \
    --predict_folder="stage1_test" \
    --label_folder="stage1_labels" \
    --label_list="train.csv" \
    --train_folder="stage1_train" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --submission_list="sample_submission.csv"