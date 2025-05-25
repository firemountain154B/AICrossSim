# =================================================================================================
# Cache directory for storing the model
export CACHE_DIR="/data/models"
export DATA_CACHE_DIR="/data/datasets"


# Output_dir of the finetuned model
export prefix="/home/thw20/projects/AICrossSim"

# Mpdel config 
# export MODEL_NAME_PREFIX="JeremiahZ/roberta-base-"
export MODEL_NAME_PREFIX="JeffreyWong/roberta-base-relu-"

# Evaluation parameters
export EVAL_BATCH_SIZE=4

# single task or multi-task
export USE_SINGLE_TASK=true

# single task parameters
export USE_CKPT=false
export MODEL_CHPT_PATH="/home/thw20/projects/AICrossSim/ckpt/roberta/finetune/mrpc/lr_1e-5"
export USE_MASE=1
export TASK_NAME="mrpc"

# multi-task parameters
export TASK_LIST="cola stsb rte sst2 qnli"

# push to hub
export PUSH_TO_HUB=false
# =================================================================================================

if [ "$USE_SINGLE_TASK" = true ]; then
    echo "Single task finetuning"

    export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/"$TASK_NAME"/SNN"

    if [ "$USE_CKPT" = true ]; then
        export MODEL_NAME=$MODEL_CHPT_PATH
    else
        export MODEL_NAME=$MODEL_NAME_PREFIX$TASK_NAME
    fi

    echo $MODEL_NAME
    echo "Evaluating on "$TASK_NAME" and output_dir="$OUTPUT_DIR" and model_name="$MODEL_NAME



    python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
    neuro_eval/run_glue_no_trainer.py \
        --seed 42 \
        --use_mase $USE_MASE \
        --task_name $TASK_NAME \
        --max_length 512 \
        --pad_to_max_length \
        --model_name_or_path $MODEL_NAME \
        --per_device_eval_batch_size $EVAL_BATCH_SIZE \
        --trust_remote_code true\
        --checkpointing_steps epoch \
        --report_to wandb \
        --num_train_epochs 0 \
        --with_tracking

fi