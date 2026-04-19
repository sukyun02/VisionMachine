OUT_DIR=./output

python -u -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=29500\
	main.py \
	--model dhvt_tiny_cifar_patch4 \
	--input-size 32 \
	--batch-size 128 \
	--accum-steps 2 \
	--warmup-epochs 5 \
	--lr 1e-3 \
	--num_workers 8 \
	--epochs 300 \
	--data-set CIFAR \
	--smoothing 0.1 \
	--coarse-loss-weight 0.3 \
	--num-superclasses 20 \
	--sc-smoothing \
	--sc-intra-ratio 1.0 \
	--early-stop-patience 30 \
	--data-path ../data/cifar-100 \
	--output_dir $OUT_DIR "$@"

