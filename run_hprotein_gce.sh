#!/usr/bin/env bash
python3 task.py \
    --run_training="True" \
    --batch_size=32 \
    --epochs=20 \
    --gpu_count=1 \
    --run_eval="True" \
    --validation_steps=24 \
    --validation_split=3072 \
    --run_predict="True" \
    --new_model="True" \
    --model_name="model" \
    --model_folder="model" \
    --train_folder="stage1_train" \
    --label_folder="stage1_labels/train.csv" \
    --predict_folder="stage1_test" \
    --submission_folder="stage1_submit" \
    --job_id=''
