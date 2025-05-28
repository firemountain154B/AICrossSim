
CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
    --mode train \
    --save_path /data/models/cx922/resnet_eval/model_original/ \
    --epochs 300 \
    --learning_rate 0.1 \
    --batch_size 256