# CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
#     --mode train \
#     --save_path /data/models/cx922/resnet_eval/model_original/ \
#     --epochs 300 \
#     --learning_rate 0.1 \
#     --batch_size 256

# CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
#     --mode test \
#     --model_path /data/models/cx922/resnet_eval/model_best.pkl \
#     --batch_size 256


# CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
#     --mode finetune \
#     --model_path /data/models/cx922/resnet_eval/model_best.pkl \
#     --save_path /data/models/cx922/resnet_eval/model_finetuned/ \
#     --batch_size 256

CUDA_VISIBLE_DEVICES=0,1 python3 acxsearch/resnet_eval/main.py \
    --mode finetune \
    --model_path /data/models/cx922/resnet_eval/model_best.pkl \
    --save_path /data/models/cx922/resnet_eval/model_finetuned/ \
    --batch_size 256 \
    --distributed \
    --world_size 2

