
DATA_HOME="/data/cx922"
PROJECT_HOME="/home/jianyicheng/cx922/AICrossSim"

DATASETS_PATH="${DATA_HOME}/datasets"
MODELS_PATH="${DATA_HOME}/models" # for models not from huggingface

export HF_HOME="${DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${DATA_HOME}/hf_transformers"
export CACHE_DIR="${PROJECT_HOME}/.cache"

cd ${PROJECT_HOME}

pwd

CUDA_VISIBLE_DEVICES=6,7 python3 acxsearch/resnet_eval/main.py \
    --mode finetune \
    --load_path /data/models/cx922/resnet_eval/model_best.pkl \
    --save_path /data/models/cx922/resnet_eval/model_finetuned/ \
    --batch_size 256 \
    --distributed \
    --world_size 2