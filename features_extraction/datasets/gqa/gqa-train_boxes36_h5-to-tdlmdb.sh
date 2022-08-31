#!/bin/bash

basedir=/home/user/fnoora/iglue
srv=/srv/scratch1/fnoora/gqa/vg_gqa_imgfeat
H5="${srv}/vg_gqa_obj36-36.h5"
LMDB="${srv}/volta/gqa-train_boxes36.lmdb"
TEXT="${basedir}/datasets/gqa/annotations/train_target.pkl"

rm -r $LMDB

conda activate volta

python gqa_boxes36_h5-to-tdlmdb.py --h5 $H5 --lmdb $LMDB --annotation $TEXT

conda deactivate
