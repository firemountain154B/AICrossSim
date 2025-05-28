
CUDA_VISIBLE_DEVICES=2 python3 acxsearch/resnet_eval/main.py \
    --mode finetune \
    --load_path /data/models/cx922/resnet_eval/model_best.pkl \
    --save_path /data/models/cx922/resnet_eval/model_cim/ \
    --batch_size 256 \
    --cim True\
    --cim_config_path /home/cx922/AICrossSim/experiments/digital.yaml