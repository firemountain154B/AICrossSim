
eval_name="original"
target_gpu=0


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

# TASK_LIST=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
TASK_LIST=["cola"]

cd ${PROJECT_HOME}
pwd

for task_name in cola; do
    model_name=JeremiahZ/roberta-base-${task_name}
    CUDA_VISIBLE_DEVICES=$target_gpu python acxsearch/roberta_eval/run_glue_training.py \
        --model_name_or_path ${model_name} \
        --task_name ${task_name} \
        --cim True \
        --cim_config_path ${PROJECT_HOME}/experiments/${eval_name}.yaml \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${CHECKPOINT_PATH}/roberta_train/${eval_name}/$task_name
done