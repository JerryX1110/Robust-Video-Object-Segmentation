datasets="youtubevos"
config="configs.resnet101_aocnet_2"
python ../tools/eval_net.py --config ${config} --dataset ${datasets} --ckpt_step 400000 --global_chunks 16 --gpu_id 0 --mem_every 5 
