DATA_HOME="/data/cx922"
PROJECT_HOME="/home/jianyicheng/cx922/AICrossSim"

DATASETS_PATH="${DATA_HOME}/datasets"
MODELS_PATH="${DATA_HOME}/models" # for models not from huggingface

export HF_HOME="${DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${DATA_HOME}/hf_transformers"
export CACHE_DIR="${PROJECT_HOME}/.cache"

# Add custom paths to system PATH
export PYTHONPATH="${PROJECT_HOME}/acxsearch/:$PYTHONPATH"

cd ${PROJECT_HOME}
pwd

CUDA_VISIBLE_DEVICES=0,1 \
DATASET_PATH=${DATASETS_PATH} \
python3 acxsearch/resnet_eval/main.py \
    --mode train \
    --save_path ${DATA_HOME}/AICrossSim/resnet_eval/model_original \
    --epochs 300 \
    --learning_rate 0.1 \
    --batch_size 256