# ===============++ custom setting =========================== #

export GPU=0
export LEARNING_RATE=5e-6
export BATCH_SIZE=128
export REPLAY_MEM=label_priority
export DQN_TYPE=ddqn
export GLUE_DIR=data/dqn
export EPOCHS=30
export SAVE_STEPS=100000
export EPS_DECAY=2000
export CAPACITY=10000
export TARGET_UPDATE=2
export NUM_LAYERS=3

# ============================================================ #


# ========= sentence encoding module setting ================= #

#export MODEL_TYPE=bert
#export MODEL_NAME=bert-base-cased
#export LOWER_CASE=0

#export MODEL_TYPE=bert
#export MODEL_NAME=bert-base-uncased
#export LOWER_CASE=1

#export MODEL_TYPE=albert
#export MODEL_NAME=albert-large-v2
#export LOWER_CASE=1

#export MODEL_TYPE=xlnet
#export MODEL_NAME=xlnet-large-cased
#export LOWER_CASE=0

export MODEL_TYPE=roberta
export MODEL_NAME=roberta-large
export LOWER_CASE=0

# ============================================================ #


# ========== evidence encoding module setting ================ #

## T-T
export DQN_MODE=transformer  # context sub-module
export AGGREGATE=transformer # aggregation sub-module
export ID=TT

### T-A
#export DQN_MODE=transformer
#export AGGREGATE=attention
#export ID=TA

### BiLSTM-T
#export DQN_MODE=lstm
#export AGGREGATE=transformer
#export ID=LT

### BiLSTM-A
#export DQN_MODE=lstm
#export AGGREGATE=attention
#export ID=LA

# ============================================================ #


export BERT_PATH=data/bert/$MODEL_NAME
export OUTPUT_DIR=output/$MODEL_NAME/$MODEL_TYPE-$DQN_TYPE-$REPLAY_MEM-$LEARNING_RATE-$ID-$NUM_LAYERS

export CHECKPOINT=output/roberta-large/roberta-ddqn-label_priority-5e-6-TT-3/9-0-0.06968751833970252

CUDA_VISIBLE_DEVICES=$GPU python src/run_train.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $BERT_PATH \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps $SAVE_STEPS \
    --dqn_type $DQN_TYPE \
    --dqn_mode $DQN_MODE \
    --eps_decay $EPS_DECAY \
    --mem $REPLAY_MEM \
    --do_lower_case $LOWER_CASE \
    --capacity $CAPACITY \
    --target_update $TARGET_UPDATE \
    --num_layers $NUM_LAYERS \
    --proportion 1 1 1 \
    --aggregate $AGGREGATE \
    --checkpoint $CHECKPOINT \
