export GLUE_DIR=data/dqn
export BATCH_SIZE=100

#export MODEL_TYPE=bert
#export BERT_TYPE=data/bert/bert-base-cased
#export LOWER_CASE=0

#export MODEL_TYPE=bert
#export BERT_TYPE=data/bert/bert-base-uncased
#export LOWER_CASE=1

#export MODEL_TYPE=albert
#export BERT_TYPE=data/bert/albert-large-v2
#export LOWER_CASE=1

#export MODEL_TYPE=xlnet
#export BERT_TYPE=data/bert/xlnet-large-cased
#export LOWER_CASE=0

export MODEL_TYPE=roberta
export BERT_TYPE=data/bert/roberta-large
export LOWER_CASE=0

# context sub-module
#export DQN_MODE=transformer
export DQN_MODE=lstm

# aggregation sub-module
#export AGGREGATE=transformer
export AGGREGATE=attention

#export CHECKPOINT=output/roberta-large/roberta-ddqn-label_priority-5e-6-TT-3/best-checkpoint
#export CHECKPOINT=output/roberta-large/roberta-ddqn-label_priority-5e-6-TA-3/best-checkpoint
export CHECKPOINT=output/roberta-large/roberta-ddqn-label_priority-5e-6-LA-3/best-checkpoint
#export CHECKPOINT=output/roberta-large/roberta-ddqn-label_priority-5e-6-LT-3/best-checkpoint

export GPU=0
#*************************************


CUDA_VISIBLE_DEVICES=$GPU python src/run_eval.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $BERT_TYPE \
    --dqn_mode $DQN_MODE \
    --data_dir $GLUE_DIR \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --do_lower_case $LOWER_CASE \
    --aggregate $AGGREGATE \
    --checkpoint $CHECKPOINT \
    --do_eval \
