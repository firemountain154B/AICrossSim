
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

CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=${DATASETS_PATH} \
python3 acxsearch/resnet_eval/main.py \
    --mode finetune \
    --load_path ${CHECKPOINT_PATH}/resnet_eval/model_original/model_best.pkl \
    --save_path ${CHECKPOINT_PATH}/resnet_eval/model_finetuned/ \
    --batch_size 256 \
    --cim True \
    --cim_config_path ${PROJECT_HOME}/experiments/digital.yaml