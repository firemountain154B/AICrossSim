
# task list : [mnli	qnli rte sst mrpc cola qqp stsb]
DATA_PATH="/mnt/home/cx922/.cache"

if [ "$CIM_EVAL" = True ]; then
    EVAL_NAME=CIM
    echo "CIM_EVAL is True"
else
    EVAL_NAME=ORIGINAL
fi
# TASK_NAME=CIM_NO_TRAIN_EVAL
export HF_HOME="${DATA_PATH}/hf_home"
export TRANFORMERS_CACHE="${DATA_PATH}/hf_transformers"

export CACHE_DIR="${DATA_PATH}/models"
export DATA_CACHE_DIR="${DATA_PATH}/datasets"

OUTPUT_DIR="/home/cx922/AICrossSim/output"

for TASK_NAME in cola mnli mrpc qnli qqp rte sst2 stsb; do
    model_name=JeremiahZ/roberta-base-$TASK_NAME
    python3 acxsearch/cim_eval/cim_eval.py \
        --model_name_or_path $model_name \
        --cim_eval $CIM_EVAL \
        --task_name $TASK_NAME \
        --cache_dir $CACHE_DIR \
        --data_cache_dir $DATA_CACHE_DIR \
        --max_length 128 \
        --per_device_eval_batch_size 32 \
        --output_dir $OUTPUT_DIR/$EVAL_NAME/$TASK_NAME
done
# TASK_NAME=mnli
# model_name=JeremiahZ/roberta-base-$TASK_NAME
# python3 acxsearch/cim_eval/cim_eval.py \
#     --cim_eval $CIM_EVAL \
#     --model_name_or_path $model_name \
#     --task_name $TASK_NAME \
#     --cache_dir $CACHE_DIR \
#     --data_cache_dir $DATA_CACHE_DIR \
#     --max_length 128 \
#     --per_device_eval_batch_size 32 \
#     --output_dir $OUTPUT_DIR/$EVAL_NAME/$TASK_NAME