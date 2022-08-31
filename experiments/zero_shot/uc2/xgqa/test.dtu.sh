#!/bin/bash

TASK=15
LANG=$2
CODE_SWITCH_LANG=zh
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36.dtu
TRTASK=GQA
TETASK=xGQA${LANG}


TEXT_PATH=/home/user/fnoora/iglue/datasets/xGQA/annotations/few_shot/${LANG}/test.json

USE=$2

PRETRAINED_PATH=$1
PRETRAINED=${PRETRAINED_PATH}/${USE}/xgqa/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=${PRETRAINED_PATH}/${USE}/xgqa/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test


conda activate iglue

cd ../../../../volta

CUDA_VISIBLE_DEVICES=0 python eval_task.py \
  --bert_model /srv/scratch1/fnoora/iglue/huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --split test_${LANG} \
  --output_dir ${OUTPUT_DIR} --val_annotations_jsonpath ${TEXT_PATH}
python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_best.bin-/test_${LANG}_result.json \
  --truth_file $TEXT_PATH

conda deactivate
cd
