#!/usr/bin/env bash
python3 task.py \
    --run_training="True" \
    --batch_size=128 \
    --epochs=10 \
    --run_eval="True" \
    --validation_steps=128 \
    --validation_split=0.1 \
    --run_predict="True" \
    --new_model="True" \
    --model_name="model" \
    --model_folder="model" \
    --train_folder="stage1_train" \
    --label_folder="stage1_labels/train.csv" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --job_id=''
