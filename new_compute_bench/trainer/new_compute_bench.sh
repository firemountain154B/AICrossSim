
# task list : [mnli	qnli rte sst mrpc cola qqp stsb]
export TASK_NAME=$1

# export model_name=JeffreyWong/roberta-base-relu-$TASK_NAME
export model_name=JeremiahZ/roberta-base-$TASK_NAME

python run_glue.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir /home/cx922/AICrossSim/reram_result/$TASK_NAME/

  # --cache_dir /home/cx922/.cache \
  # --data_cache_dir /home/cx922/.cache \
  # --output_dir /tmp/$TASK_NAME/