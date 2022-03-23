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
class MSMARCODataset(Dataset):
        def __init__(self, queries, corpus, asym=False):
            self.queries = queries
            self.queries_ids = list(queries.keys())
            self.corpus = corpus

            self.asym = asym

            for qid in self.queries:
                #self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
                #self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
                random.shuffle(self.queries[qid]['neg'])

        def __getitem__(self, item):
            query = self.queries[self.queries_ids[item]]
            query_text = query['query']

            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)

            neg_id = query['neg'].pop(0)  # Pop negative and add at end
            neg_text = self.corpus[neg_id]
            query['neg'].append(neg_id)

            if self.asym:
                return InputExample(texts=[{'QRY': query_text}, {'DOCPOS': pos_text}, {'DOCNEG': neg_text}])

            return InputExample(texts=[query_text, pos_text, neg_text])

        def __len__(self):
            return len(self.queries)

parser = argparse.ArgumentParser()
                    # these arguments are used for global path locating
parser.add_argument("--identifier", default="sgpt-125M-msmarco", type=str)
parser.add_argument("--cache_folder", default="/data/private/huxiaomeng/pretrained_models", type=str)
parser.add_argument("--data_folder", default="/data/private/huxiaomeng/sgpt/", type=str)
parser.add_argument("--checkpoint_save_folder", default="datasets/msmarco/", type=str)
#parser.add_argument("--results_save_folder", default="datasets/msmarco/", type=str)
#parser.add_argument("--model_identifier", default="checkpoints/msmarco/", type=str)
                    # these arguments are used for saving intermediate results
#parser.add_argument("--dev_corpus_embedding_path", default="embeddings/dev/", type=str,
#                    help="save dev_corpus's embedding")
#parser.add_argument("--dev_topk_score_path", default="topk_score/dev/", type=str,
#                    help="save topk scores of dev_queries and dev_corpus")
#parser.add_argument("--test_corpus_embedding_path", default="embeddings/test/", type=str,
#                    help="save test_corpus's embedding")
#parser.add_argument("--test_topk_score_path", default="topk_score/test/", type=str,
#                    help="save topk scores of test_queries and test_corpus")
                    # these arguments are used for loading dataset
parser.add_argument("--corpus_name", default="collection.tsv", type=str)
parser.add_argument("--train_queries_name", default="queries.tsv", type=str)
parser.add_argument("--train_qrels_name", default="qrels.tsv", type=str)
#parser.add_argument("--dev_queries_name", default="queries.tsv", type=str)
#parser.add_argument("--dev_qrels_name", default="qrels.tsv", type=str)
#parser.add_argument("--test_queries_name", default="queries.tsv", type=str)
#parser.add_argument("--test_qrels_name", default="qrels.tsv", type=str)
                    # these arguments are used for training or inferencing
parser.add_argument("--train_batch_size", default=64, type=int)
#parser.add_argument("--encode_batch_size", default=64, type=int,
#                    help="batch to encode corpus or queries during inference")
#parser.add_argument("--corpus_chunk_size", default=64, type=int,
#                    help="split the corpus into several chunks to degrade the computation complexity")
                    # these arguments are used loading or creating a model
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument("--freezenonbias", action="store_true", help="Freeze all except biases in transformer")
parser.add_argument("--unfreezewte", action="store_true", help="Unfreeze Word Token Embeddings")
                    # decides train of inference
#parser.add_argument("--no_training", action="store_true")
                    # training settings
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--steps_per_epoch", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--negs_to_use", default=None,
                    help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--train_dataset_max_size", default=None, type=int)
parser.add_argument("--dev_corpus_max_size", default=-1, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--seed", default=13, type=int)
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandbwatchlog", default="all", type=str) # Set e.g. to just gradients for large models
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

print(args)

data_folder = args.data_folder

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# The  model we want to fine-tune
model_name_or_path = args.model_name_or_path

train_batch_size = args.train_batch_size  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train
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

if args.freeze or args.freezenonbias:
    for name, param in model.named_parameters():
        if args.freezenonbias and "bias" in name:
            # Freeze all except bias
            continue 
        if args.unfreezewte and "wte" in name:
            # Do not freeze Word Token Embeddings
            continue
        param.requires_grad = False

collection_filepath = os.path.join(data_folder, args.corpus_name)
train_queries_filepath = os.path.join(data_folder, args.train_queries_name)
#dev_queries_filepath=os.path.join(data_folder, args.dev_queries_name)
#test_queries_filepath=os.path.join(data_folder, args.test_queries_name)
train_qrels_filepath=os.path.join(data_folder, args.train_qrels_name)
#dev_qrels_filepath=os.path.join(data_folder, args.dev_qrels_name)
#test_qrels_filepath=os.path.join(data_folder, args.test_qrels_name)
                ## only training needs the two files
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')

corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
queries = {}  
#dev_queries = {}
train_rel_docs={}
#dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need

with open(train_queries_filepath, 'r', encoding='utf8') as fIn:
    for idx, line in enumerate(fIn):
        qid, query = line.strip().split("\t")
       # qid = int(qid)
        if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
            query="Query: "+query
        else :
            query="Query: "+query
            query = "[SOS]" + query
        if idx == 0:
            logging.info(f"Train Query Example: {query}")
        queries[qid] = query


###     Read train relevant documents
#with open(train_qrels_filepath) as fIn:
#    for line in fIn:
#        qid, _, pid, score = line.strip().split('\t')
#        if eval(score) < 1:
#            continue
#        if qid not in train_queries:
#            continue
#
#        if qid not in train_rel_docs:
#            train_rel_docs[qid] = set()
#        train_rel_docs[qid].add(pid)
#
#        needed_pids.add(pid)

# Read dev passages
logging.info("Read corpus: collection.tsv")
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
            pid,title,body =line.strip().split('\t')
            title,body=title.strip(),body.strip()
            passage="Title: "+title+"Passage: "+body
        else:
            pid,title,body =line.strip().split('\t')
            title,body=title.strip(),body.strip()
            passage="Title: "+title+"Passage: "+body
            passage = "{SOS}" + passage
        corpus[pid] = passage

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for i, line in tqdm.tqdm(enumerate(fIn)):
        data = json.loads(line)

        # Get the positive passage ids
        qid =data['qid']
        pos_pids =data['pos']

        if len(pos_pids) == 0:  # Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:  # Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:  # Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(str(pid))
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if str(data['qid']) in queries and len(neg_pids)>0:
            train_queries[str(data['qid'])] = {'qid': str(data['qid']), 'query': queries[str(data['qid'])], 'pos':[str(id) for id in pos_pids],
                                            'neg': list(neg_pids)}

        if args.train_dataset_max_size is not None and i > args.train_dataset_max_size:
            break

logging.info("Train queries: {}".format(len(train_queries)))

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
#dev_dataset = MSMARCODataset(dev_queries, corpus=corpus, asym=args.asym)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
#dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=train_batch_size*4 if args.dev_batch_size is None else args.dev_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
if args.wandb and accelerator.is_main_process:
    wandb.watch(model, log=args.wandbwatchlog, criterion=train_loss, log_freq=100)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=args.warmup_steps,
            use_amp=args.use_amp,
            checkpoint_save_folder=args.checkpoint_save_folder,
            optimizer_params={'lr': args.lr},
            show_progress_bar=True,
            steps_per_epoch=args.steps_per_epoch,
            accelerator=accelerator
            )
