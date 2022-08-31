#!/bin/bash

INDIR=/srv/scratch1/fnoora/iglue/gqa/features/vg-gqa_X101.npy
OUTDIR=/srv/scratch1/fnoora/iglue/gqa/features/vg-gqa_X101.lmdb

mkdir -p $OUTDIR

conda activate iglue

python ../../npy_to_lmdb.py --mode convert --features_folder $INDIR --lmdb_path $OUTDIR

conda deactivate
