#!/usr/bin/env bash
python3 task.py \
    --run_training="False" \
    --batch_size=128 \
    --epochs=100 \
    --gpu_count=8 \
    --run_eval="True" \
    --validation_steps=24 \
    --validation_split=3072 \
    --run_predict="True" \
    --new_model="False" \
    --model_name="model_365690" \
    --model_folder="model" \
    --train_folder="stage1_train" \
    --label_folder="stage1_labels/train.csv" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --job_id=''
