#!/bin/bash

TASK=15
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_trainval_tasks_boxes36.dtu

PRETRAINED_PATH=$2
PRETRAINED=${PRETRAINED_PATH}/${MODEL}/${MODEL_CONFIG}/uc2_checkpoint_200000.bin

FINE_TUNED_MODEL=${MODEL}_zero_shot_semanticloss_seed_${1}

OUTPUT_DIR=${3}/${FINE_TUNED_MODEL}/xgqa/${MODEL}
LOGGING_DIR=${3}/${FINE_TUNED_MODEL}/xgqa/${MODEL_CONFIG}

conda activate volta

cd ../../../../volta

CUDA_VISIBLE_DEVICES=0 python train_task.py \
  --bert_model  /srv/scratch1/fnoora/iglue/huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} --cache 500 \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --gradient_accumulation_steps 4 --num_workers 20 --num_val_workers 20 \
  --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --drop_last \
  --seed $1 \

conda deactivate
cd

