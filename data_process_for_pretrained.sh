#!/bin/bash
python src/pretrain_model/convert_to_glue_input.py \
    -s data/retrieved/train.ensembles.s10.jsonl \
    -o data/glue/train.tsv
python src/pretrain_model/convert_to_glue_input.py \
    -s data/retrieved/dev.ensembles.s10.jsonl \
    -o data/glue/dev_matched.tsv
python src/pretrain_model/convert_to_glue_input.py \
    -s data/retrieved/dev.ensembles.s10.jsonl \
    -o data/glue/dev_mismatched.tsv
