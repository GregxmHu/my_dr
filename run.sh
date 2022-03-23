#model_name=/data/private/huxiaomeng/pretrained_models/SGPT-5.8B-weightedmean-msmarco-specb-bitfit
model_name=Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 2222 examples/training/ms_marco/train_bi-encoder_mnrl.py --use_pre_trained_model --model_name ${model_name}  --train_batch_size 2    --eval_batch_size 100 --freezenonbias --specb --lr 0.5e-4 --wandb --wandbwatchlog gradients --pooling weightedmean --epochs 0
