set -ex
export OMP_NUM_THREADS=1
datasets=$3
model=$1
pooling=$2
identifier="${model}_${pooling}-pooling_${datasets}"
project_path="/data/home/scv0540/run/my_dr"
cache_folder="/data/home/scv0540/run/pretrained_models/"
data_folder="${project_path}/datasets/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
corpus_name="corpus_with_title.tsv"
train_queries_name="queries.train.tsv"
train_qrels_name="qrels-irrels.train.tsv"
model_name_or_path="${cache_folder}/t5-base-scaled"
log_dir="$project_path/logs/$identifier"
#model_name_or_path="/data/home/scv0540/run/my_dr/checkpoints/t5-base-scaled_cls-pooling_msmarco/epoch_15"
accelerate launch\
 --config_file accelerate_config.yaml\
 examples/training/ms_marco/train_msmarco.py\
 --identifier $identifier\
 --cache_folder $cache_folder\
 --data_folder $data_folder\
 --checkpoint_save_folder $checkpoint_save_folder\
 --corpus_name $corpus_name\
 --train_queries_name $train_queries_name\
 --train_qrels_name $train_qrels_name\
 --model_name_or_path ${model_name_or_path}\
 --train_batch_size 32\
 --pooling $pooling\
 --epochs 5\
 --use_amp\
 --bootstrap\
 --log_dir=$log_dir\
 --round 2\