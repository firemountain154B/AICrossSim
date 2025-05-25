
for task in stsb qnli mrpc cola qqp sst2 mnli rte
do
    CUDA_VISIBLE_DEVICES=2,3 sh new_compute_bench.sh $task
done