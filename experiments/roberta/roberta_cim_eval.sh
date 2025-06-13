
# eval_name="original"
target_gpu=6


DATA_HOME="/data/cx922"
PROJECT_HOME="/home/jianyicheng/cx922/AICrossSim"

DATASETS_PATH="${DATA_HOME}/datasets"
MODELS_PATH="${DATA_HOME}/models" # for models not from huggingface

CHECKPOINT_PATH="${DATA_HOME}/AICrossSim"

export HF_HOME="${DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${DATA_HOME}/hf_transformers"
export CACHE_DIR="${PROJECT_HOME}/.cache"

# Add custom paths to system PATH
export PYTHONPATH="${PROJECT_HOME}/acxsearch/:$PYTHONPATH"

cd ${PROJECT_HOME}
pwd


for eval_name in sram pcm reram; do
    for task_name in cola mnli mrpc qnli qqp rte sst2 stsb; do
        model_name=JeremiahZ/roberta-base-${task_name}
        CUDA_VISIBLE_DEVICES=$target_gpu python3 acxsearch/roberta_eval/run_gelu.py \
            --model_name_or_path ${model_name} \
            --cim True \
            --cim_config_path ${PROJECT_HOME}/experiments/${eval_name}.yaml \
            --task_name ${task_name} \
            --data_cache_dir ${DATASETS_PATH} \
            --max_length 128 \
            --per_device_eval_batch_size 32 \
            --output_dir ${CHECKPOINT_PATH}/roberta_eval/${eval_name}/$task_name
    done
done