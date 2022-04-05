set -ex
export OMP_NUM_THREADS=1

model=$1
pooling=$2
datasets=$3
round=$4
stage=$5

test_queries_name="queries.dev.small.tsv"
test_qrels_name="qrels.dev.small.tsv"
train_queries_name="queries.train.tsv"
train_qrels_name="qrels.train.tsv"
corpus_name="corpus_with_title.tsv"

identifier="${model}_${pooling}-pooling_${datasets}"
project_path="/data/home/scv0540/run/my_dr"
cache_folder="/data/home/scv0540/run/pretrained_models/"
data_folder="${project_path}/datasets/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
model_name_or_path="${checkpoint_save_folder}/round${round}-stage${stage}/"

results_save_folder="${project_path}/results/${identifier}/"
test_topk_score_path="${project_path}/scores/${identifier}_test.tsv"
train_topk_score_path="${project_path}/scores/${identifier}_train.tsv"

accelerate launch\
 --config_file accelerate_config.yaml\
 inference_v2.py\
 --identifier $identifier\
 --cache_folder $cache_folder\
 --data_folder $data_folder\
 --results_save_folder $results_save_folder\
 --test_topk_score_path $test_topk_score_path\
 --test_queries_name $test_queries_name\
 --test_qrels_name $test_qrels_name\
 --train_topk_score_path $train_topk_score_path\
 --train_queries_name $train_queries_name\
 --train_qrels_name $train_qrels_name\
 --corpus_name $corpus_name\
 --encode_batch_size 50\
 --corpus_chunk_size 10000\
 --model_name_or_path ${model_name_or_path}\
 --pooling $pooling\
 --round $round\
 --stage $stage\
 --seed 13\
 --use_pre_trained_model\

