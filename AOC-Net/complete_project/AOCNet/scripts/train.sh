datasets="youtubevos"
config="configs.resnet101_aoc"
# training for 200k with lr=0.2
python ../tools/train_net_mm.py --config ${config} --datasets ${datasets}  --global_chunks 1

# go on training for 200k with lr=0.1
config="configs.resnet101_aoc_2"
python ../tools/train_net_mm.py --config ${config} --datasets ${datasets}  --global_chunks 1
