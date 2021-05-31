#*************************************

#export MODEL_TYPE=bert
#export MODEL_NAME=bert-base-cased
#export LEARNING_RATE=2e-5
#export LOWER_CASE=0

#export MODEL_TYPE=bert
#export MODEL_NAME=bert-base-uncased
#export LOWER_CASE=1
#export BATCH_SIZE=12

#export MODEL_TYPE=xlnet
#export MODEL_NAME=xlnet-large-cased
#export LOWER_CASE=0
#export BATCH_SIZE=4
 
#export MODEL_TYPE=albert
#export MODEL_NAME=albert-large-v2
#export LOWER_CASE=1
#export BATCH_SIZE=8

export MODEL_TYPE=roberta
export MODEL_NAME=roberta-large
export LOWER_CASE=0
export BATCH_SIZE=4

export CHECKPOINT=data/bert/$MODEL_NAME

export GPU=0
#*************************************
#export LEARNING_RATE=5e-6
export BERT_PATH=data/bert/$MODEL_NAME
export OUTPUT_DIR=output/$MODEL_NAME/$MODEL_TYPE-pretrained
export GLUE_DIR=data/glue/
export MAX_SEQ_LENGTH=256
export EPOCHS=5
export GRADIENT_STPES=1
export LEARNING_RATE=2e-6

CUDA_VISIBLE_DEVICES=$GPU python src/pretrain_model/run_glue.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $CHECKPOINT \
    --config_name $BERT_PATH \
    --tokenizer_name $BERT_PATH \
    --task_name mnli \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --do_lower_case $LOWER_CASE \
    --save_steps 2000000000 \
    --gradient_accumulation_steps $GRADIENT_STPES
