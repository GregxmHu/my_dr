set -ex
export OMP_NUM_THREADS=1
datasets=$3
model=$1
pooling=$2
round_idx=$4
stage_idx=$5
identifier="${model}_${pooling}-pooling_${datasets}"
project_path="/data/home/scv0540/run/my_dr"
cache_folder="/data/home/scv0540/run/pretrained_models/"
data_folder="${project_path}/datasets/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
corpus_name="corpus_with_title.tsv"
train_queries_name="queries.train.tsv"
train_qrels_name="with.hard.qrels-irrels.train.tsv"
log_dir="$project_path/logs/$identifier"
base_model_name_or_path="${cache_folder}/gpt-125m"
model_name_or_path="${checkpoint_save_folder}/round${round_idx}-stage${stage_idx}/"
accelerate launch\
 --config_file accelerate_config.yaml\
 train_msmarco_v2.py\
 --identifier $identifier\
 --cache_folder $cache_folder\
 --data_folder $data_folder\
 --checkpoint_save_folder $checkpoint_save_folder\
 --corpus_name $corpus_name\
 --train_queries_name $train_queries_name\
 --train_qrels_name $train_qrels_name\
 --model_name_or_path ${model_name_or_path}\
 --base_model_name_or_path ${base_model_name_or_path}\
 --train_batch_size 32\
 --pooling $pooling\
 --epochs 5\
 --use_amp\
 --bootstrap\
 --log_dir=$log_dir\
 --round_idx $round_idx\
 --stage_idx $stage_idx\
 --use_pre_trained_model\
 --freezenonbias\
