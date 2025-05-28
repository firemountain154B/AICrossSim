
eval_name="original"
target_gpu=0
output_dir="/home/cx922/AICrossSim/.cache/"
# ----------------------------------------------------------
# ------ normally you need to give modification here -------
# ----------------------------------------------------------

# task list : [mnli	qnli rte sst mrpc cola qqp stsb]
DATA_HOME="/data/models/cx922"
PROJECT_HOME="/home/cx922/AICrossSim"

DATASETS_PATH="${DATA_HOME}/datasets"
MODELS_PATH="${DATA_HOME}/models" # for models not from huggingface

export HF_HOME="${DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${DATA_HOME}/hf_transformers"
export CACHE_DIR="${PROJECT_HOME}/.cache"

for TASK_NAME in cola mnli mrpc qnli qqp rte sst2 stsb; do
    model_name=JeremiahZ/roberta-base-$TASK_NAME
    CUDA_VISIBLE_DEVICES=$target_gpu python3 acxsearch/roberta_eval/run_gelu.py \
        --model_name_or_path $model_name \
        --task_name $TASK_NAME \
        --data_cache_dir $DATASETS_PATH \
        --max_length 128 \
        --per_device_eval_batch_size 32 \
        --output_dir $output_dir/$eval_name/$TASK_NAME
done