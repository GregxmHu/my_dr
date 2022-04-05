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
from tqdm.autonotebook import trange
from torch.utils.tensorboard import SummaryWriter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch.cuda
import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import time
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
def getdata(train_queries_filepath,collection_filepath,train_qrels_filepath):
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
    train_queries = {}  
    with open(train_qrels_filepath) as fIn:
        for line in fIn:
            qid, pos_pids, neg_pids = line.strip('\n').split('\t')
            train_queries[qid]={'qid':qid,'pos':eval(pos_pids),'neg':eval(neg_pids)}

    with open(train_queries_filepath, 'r', encoding='utf8') as fIn:
        for idx, line in enumerate(fIn):
            qid, query = line.strip().split("\t")
            if qid not in train_queries:
                continue
        # qid = int(qid)
            if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
                query="Query: "+query
            else :
                query="Query: "+query
                query = "[SOS]" + query
            if idx == 0:
                logging.info(f"Train Query Example: {query}")
            train_queries[qid]['query'] = query



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

    return train_queries,corpus



class MSMARCODataset(Dataset):
        def __init__(self, queries, corpus, asym=False):
            self.queries = queries
            self.queries_ids = list(queries.keys())
            self.corpus = corpus

            self.asym = asym

            #for qid in self.queries:
                #self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
                #self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            #    random.shuffle(self.queries[qid]['neg'])

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
parser.add_argument("--log_dir", type=str,default=None)
                    # decides train of inference
#parser.add_argument("--no_training", action="store_true")
                    # training settings
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--steps_per_epoch", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
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
parser.add_argument("--bootstrap", default=False,action="store_true")
parser.add_argument("--round", default=1,type=int)
args = parser.parse_args()


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

#if args.log_dir is not None and accelerator.is_main_process:
#        writer = SummaryWriter(args.log_dir)
#        tb = writer
#else:
#    tb = None

print(args)
#if args.wandb and accelerator.is_main_process:
#    import wandb
#    wandb.init(project="sgpt", entity="muennighoff")
#    wandb.config.update(args)

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
train_qrels_filepath=os.path.join(data_folder, args.train_qrels_name)

train_queries,corpus=getdata(train_queries_filepath,collection_filepath,train_qrels_filepath)
logging.info("Train queries: {}".format(len(train_queries)))

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

qrels_path_list=[]
qrels_dir=os.path.join(data_folder,args.identifier)
qrels_path_list_file=os.path.join(qrels_dir,"qrels_path.tsv" )
#fp=open(qrels_path_list_file,'w')
# Train the model
if args.bootstrap:
    for round_idx in trange(args.round,desc="Bootstrap Round", disable=not accelerator.is_main_process):
        for stage in trange(2,desc="Stage", disable=not accelerator.is_main_process):
            if args.log_dir is not None and accelerator.is_main_process:
                writer = SummaryWriter(os.path.join(args.log_dir,"round{}-stage{}".format(round_idx,stage)))
                tb = writer
            else:
                tb=None
            # first fit
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_epochs,
                    warmup_steps=args.warmup_steps,
                    use_amp=args.use_amp,
                    checkpoint_save_folder=args.checkpoint_save_folder,
                    optimizer_params={'lr': args.lr},
                    show_progress_bar=accelerator.is_main_process,
                    steps_per_epoch=args.steps_per_epoch,
                    accelerator=accelerator,
                    tb=tb,
                    round=round_idx,
                    stage=stage
                    )
            # wite for new qrels to be generated
            while (round_idx,stage) not in qrels_path_list:
                print("******waiting new qrels to be generated by inference process...******")
                with open(qrels_path_list_file,'r') as f:
                    for item in f:
                        round_num,stage_num,path=item.strip('\n').split('\t')
                        round_num,stage_num=int(round_num),int(stage_num)
                        if (round_num,stage_num) not in qrels_path_list:
                            qrels_path_list.append((round_num,stage_num))
                time.sleep(60)
            # use new qrels create new datasets
            train_qrels_filepath=os.path.join(qrels_dir, "round{}-stage{}_".format(round_idx,stage)+args.train_qrels_name)
            train_queries,corpus=getdata(train_queries_filepath,collection_filepath,train_qrels_filepath)
            logging.info("Train queries: {}".format(len(train_queries)))
            train_dataset = MSMARCODataset(train_queries, corpus=corpus)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
else:
    if args.log_dir is not None and accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(args.log_dir,"round{}-stage{}".format(0,0)))
        tb = writer
    else:
        tb=None
    model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=args.warmup_steps,
                use_amp=args.use_amp,
                checkpoint_save_folder=args.checkpoint_save_folder,
                optimizer_params={'lr': args.lr},
                show_progress_bar=True,
                steps_per_epoch=args.steps_per_epoch,
                accelerator=accelerator,
                tb=tb,
                round=0,
                stage=0
                )

