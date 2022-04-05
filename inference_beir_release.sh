set -ex
export OMP_NUM_THREADS=1

#model=$1
pooling="weightedmean"
#datasets=$3
round=5
stage=5
beir_datasets=$1

test_queries_name="queries.jsonl"
test_qrels_name="qrels.tsv"
train_queries_name="queries.jsonl"
train_qrels_name="qrels.tsv"
corpus_name="corpus.jsonl"

identifier="released_gpt125M"
project_path="/data/home/scv0540/run/my_dr"
cache_folder="/data/home/scv0540/run/pretrained_models/"
data_folder="${project_path}/datasets/${beir_datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
#model_name_or_path="${checkpoint_save_folder}/round${round}-stage${stage}/"
model_name_or_path="/data/home/scv0540/run/pretrained_models/sgpt-125M"
results_save_folder="${project_path}/results/${beir_datasets}/${identifier}/"
test_topk_score_path="${project_path}/scores/${beir_datasets}/${identifier}_test.tsv"
train_topk_score_path="${project_path}/scores/${beir_datasets}/${identifier}_train.tsv"

accelerate launch\
 --config_file accelerate_config.yaml\
 inference_beir_v2.py\
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
 --encode_batch_size 10\
 --corpus_chunk_size 300\
 --model_name_or_path ${model_name_or_path}\
 --pooling $pooling\
 --round $round\
 --stage $stage\
 --seed 13\
 --use_pre_trained_model\

