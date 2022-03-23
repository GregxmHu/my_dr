set -ex
#export CUDA_VISIBLE_DEVICES=0,3
export OMP_NUM_THREADS=1
#accelerate config
model=$1
pooling=$2
datasets=$3
epoch=$4
identifier="${model}_${pooling}-pooling_${datasets}"
project_path="/data/home/scv0540/run/my_dr"
cache_folder="/data/home/scv0540/run/pretrained_models/"
data_folder="${project_path}/datasets/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
model_name_or_path="${checkpoint_save_folder}/epoch_${epoch}/"
results_save_folder="${project_path}/results/${identifier}/"
#dev_corpus_embedding_path="${project_path}/corpus_embeddings/${identifier}_dev.txt"
test_corpus_embedding_path="${project_path}/corpus_embeddings/${identifier}_test.txt"
#dev_topk_score_path="${project_path}/scores/${identifier}_dev.txt"
test_topk_score_path="${project_path}/scores/${identifier}_test.txt"
corpus_name="corpus_with_title.tsv"
test_queries_name="queries.dev.small.tsv"
test_qrels_name="qrels.dev.small.tsv"
accelerate launch\
 --config_file accelerate_config.yaml\
 examples/training/ms_marco/inference.py\
 --identifier $identifier\
 --cache_folder $cache_folder\
 --data_folder $data_folder\
 --results_save_folder $results_save_folder\
 --test_corpus_embedding_path $test_corpus_embedding_path\
 --test_topk_score_path $test_topk_score_path\
 --test_queries_name $test_queries_name\
 --test_qrels_name $test_qrels_name\
 --corpus_name $corpus_name\
 --encode_batch_size 300\
 --corpus_chunk_size 200000\
 --model_name_or_path ${model_name_or_path}\
 --pooling $pooling\
 --epoch $epoch\
 --seed 13\
 --use_pre_trained_model\
