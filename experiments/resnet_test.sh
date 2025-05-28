
CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
    --mode test \
    --model_path /data/models/cx922/resnet_eval/model_best.pkl \
    --batch_size 256
