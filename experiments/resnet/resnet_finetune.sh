
CUDA_VISIBLE_DEVICES=0,1 python3 acxsearch/resnet_eval/main.py \
    --mode finetune \
    --load_path /data/models/cx922/resnet_eval/model_best.pkl \
    --save_path /data/models/cx922/resnet_eval/model_finetuned/ \
    --batch_size 256 \
    --distributed \
    --world_size 2