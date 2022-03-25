"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch.cuda
import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--identifier", default="sgpt-125M-msmarco", type=str)
parser.add_argument("--cache_folder", default="/data/private/huxiaomeng/pretrained_models", type=str)
parser.add_argument("--data_folder", default="/data/private/huxiaomeng/sgpt/", type=str)
parser.add_argument("--results_save_folder", default="datasets/msmarco/", type=str)
parser.add_argument("--test_corpus_embedding_path", default="embeddings/test/", type=str,
                    help="save test_corpus's embedding")
parser.add_argument("--test_topk_score_path", default="topk_score/test/", type=str,
                    help="save topk scores of test_queries and test_corpus")
parser.add_argument("--corpus_name", default="collection.tsv", type=str)
parser.add_argument("--test_queries_name", default="queries.tsv", type=str)
parser.add_argument("--test_qrels_name", default="qrels.tsv", type=str)
parser.add_argument("--encode_batch_size", default=64, type=int,
                    help="batch to encode corpus or queries during inference")
parser.add_argument("--corpus_chunk_size", default=64, type=int,
                    help="split the corpus into several chunks to degrade the computation complexity")
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--max_seq_length", default=400, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandbwatchlog", default="all", type=str) # Set e.g. to just gradients for large models
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--round",type=int,default=1)
parser.add_argument("--stage",type=int,default=1)
parser.add_argument("--seed",type=int,default=1)
args = parser.parse_args()

print(args)
if not os.path.exists(args.results_save_folder):
    os.mkdir(args.results_save_folder)
if os.path.exists(args.test_corpus_embedding_path):
    with open(args.test_corpus_embedding_path,'w'):
        pass
if os.path.exists(args.test_topk_score_path):
    with open(args.test_topk_score_path,'w'):
        pass
data_folder = args.data_folder

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# The  model we want to fine-tune
model_name_or_path = args.model_name_or_path

#train_batch_size = args.train_batch_size  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
#ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
#num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
#num_epochs = args.epochs  # Number of epochs we want to train

if "gpt" in model_name_or_path or "GPT" in model_name_or_path:
    accelerator = Accelerator()
else:
    # Needed to run e.g. bert-large-uncased (Can also be used with GPT but will use unnecessary memory)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

if args.wandb and accelerator.is_main_process:
    import wandb
    wandb.init(project="sgpt", entity="muennighoff")
    wandb.config.update(args)

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained Sentence-Transformer model")
    model = SentenceTransformer(model_name_or_path,cache_folder=args.cache_folder)
    model.max_seq_length = max_seq_length
    if "gpt" in model_name_or_path or "GPT" in model_name_or_path:

        word_embedding_model = model._first_module()
        assert isinstance(word_embedding_model, models.Transformer)

        tokens = ["[SOS]", "{SOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        # Will be replaced with the rep ones
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]

        word_embedding_model.bos_spec_token_q_rep = word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]
        
        word_embedding_model.bos_spec_token_d_rep = word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]

        word_embedding_model.replace_bos = True
else:
    logging.info("Create new Sentence-Transformer model")
    word_embedding_model = models.Transformer(model_name_or_path, max_seq_length=max_seq_length,cache_folder=args.cache_folder)
    if "gpt" in model_name_or_path or "GPT" in model_name_or_path:
        word_embedding_model.tokenizer.pad_token = word_embedding_model.tokenizer.eos_token     
        tokens = ["[SOS]", "{SOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]

        word_embedding_model.bos_spec_token_q_rep = word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]
        
        word_embedding_model.bos_spec_token_d_rep = word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]

        word_embedding_model.replace_bos = True

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(cache_folder=args.cache_folder,modules=[word_embedding_model, pooling_model])
    model.max_seq_length = max_seq_length
# Evaluate
### Load eval data
collection_filepath = os.path.join(data_folder, args.corpus_name)
test_queries_file = os.path.join(data_folder, args.test_queries_name)
test_qrels_filepath = os.path.join(data_folder, args.test_qrels_name)

corpus = {}  # Our corpus pid => passage
test_queries = {}  # Our dev queries. qid => query
test_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

### Download files if needed
# Load the 6980 dev queries
with open(test_queries_file, encoding='utf8') as fIn:
    for idx,line in enumerate(fIn):
        qid, query = line.strip().split("\t")
        #qid = int(qid)
        if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
            query="Query: "+query
        else :
            query="Query: "+query
            query = "[SOS]" + query
        if idx == 0:
            logging.info(f"Train Query Example: {query}")
        test_queries[qid] = query

# Load which passages are relevant for which queries
with open(test_qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')
        if qid not in test_queries:
            continue

        if qid not in test_rel_docs:
            test_rel_docs[qid] = set()
        test_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)
# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
            pid,title,body =line.strip().split('\t')
            title,body=title.strip(),body.strip()
            passage="Title: "+title+"Passage: "+body
        else:
            if 1>2:
                pid,passage=line.strip().split('\t')
                pid,passage=pid.strip(),passage.strip()
                passage = "{SOS}" + passage
            else:
                pid,title,body =line.strip().split('\t')
                title,body=title.strip(),body.strip()
                passage="Title: "+title+"Passage: "+body
                passage = "{SOS}" + passage
        corpus[pid] = passage


model=accelerator.prepare(model)
model.eval()
# only performing evaluation from one process
ir_evaluator = evaluation.InformationRetrievalEvaluator(test_queries, corpus, test_rel_docs,
                                                        show_progress_bar=True,
                                                        corpus_chunk_size=args.corpus_chunk_size,
                                                        ndcg_at_k=[10],
                                                        encode_batch_size=args.encode_batch_size,
                                                        name="test",
                                                        accelerator=accelerator,
                                                        results_save_folder=args.results_save_folder,
                                                        corpus_embedding_path=args.test_corpus_embedding_path,
                                                        score_path=args.test_topk_score_path
                                                        )
ir_evaluator(model,round=args.round,stage=args.stage)
