from multiprocessing.dummy import current_process
from . import SentenceEvaluator
import torch
from torch import Tensor
import logging
from tqdm import tqdm, trange
from ..util import cos_sim, dot_score
import os
import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from typing import List, Tuple, Dict, Set, Callable
import json
class Corpus(Dataset):
        def __init__(self, corpus):
            self.corpus_ids = list(corpus.keys())
            self.corpus=corpus

        def __getitem__(self, idx):
            return {'corpus_id':self.corpus_ids[idx],
            'corpus_text':self.corpus[
                self.corpus_ids[idx]
                ]
                }

        def __len__(self):
            return len(self.corpus_ids)

        def collate(self,batch):
            batch_ids=[item['corpus_id'] for item in batch]
            batch_texts=[item['corpus_text'] for item in batch]
            return {'batch_ids':batch_ids,'batch_texts':batch_texts}


logger = logging.getLogger(__name__)

class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 320,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [10],
                 precision_recall_at_k: List[int] = [10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 encode_batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim, 'dot_score': dot_score},       #Score function, higher=more similar
                 main_score_function: str = None,
                 accelerator: Accelerator = None,
                 corpus_embedding_path: str="/data/private/huxiaomeng/sgpt/corpus_embeddings/msmarco_small.txt",
                 results_save_folder: str=None,
                 score_path: str="/data/private/huxiaomeng/sgpt/corpus_embeddings/sgpt-125M-scifact_topk_score.txt",
                 ):
        self.score_path=score_path
        self.corpus_chunk_size=corpus_chunk_size
        self.encode_batch_size=encode_batch_size
        self.corpus_embedding_path=corpus_embedding_path
        self.results_save_folder=results_save_folder
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]
        
        # self.corpus_ids = list(corpus.keys())
        # self.corpus = [corpus[cid] for cid in self.corpus_ids]
        ###
        self.accelerator=accelerator
        cids=list(corpus.keys())
        self.corpus_ids=[cids[idx] for idx in range(len(cids)) if idx % accelerator.num_processes == accelerator.process_index]
        self.corpus=[corpus[idx] for idx in self.corpus_ids]
        #self.corpus_dataset=Corpus(corpus)
        #self.corpus_dataloader=accelerator.prepare(
        #    DataLoader(self.corpus_dataset,batch_size=self.corpus_chunk_size,collate_fn=self.corpus_dataset.collate,num_workers=12)
        #)
        ###
        self.relevant_docs = relevant_docs
        #self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.encode_batch_size = encode_batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = name + "_results.csv"
        self.csv_headers = ["round","stage"]

        for score_name in self.score_function_names:
            #for k in accuracy_at_k:
            #    self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            #for k in precision_recall_at_k:
            #    self.csv_headers.append("{}-Precision@{}".format(score_name, k))
            #    self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            #for k in map_at_k:
            #    self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model,  round: int = 1,stage: int =1,  num_proc: int = None, *args, **kwargs) -> float:
        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + " after round {} stage {}".format(round,stage))

        self.compute_metrices(model,round=round,stage=stage, *args, num_proc=num_proc, **kwargs)
        #return score

        #if self.main_score_function is None:
        #    return max([scores[name]['map@k'][max(self.map_at_k)] for name in self.score_function_names])
        #else:
        #    return scores[self.main_score_function]['map@k'][max(self.map_at_k)]

    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None, num_proc: int = None,round: int = 1,stage: int =1) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model
        #model,corpus_model=self.accelerator.prepare(model,corpus_model)
        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))
        query_embeddings = model.module.encode(self.queries, show_progress_bar=True, batch_size=self.encode_batch_size, convert_to_tensor=True, num_proc=num_proc,accelerator=self.accelerator)
        queries_result_list = {}
        logger.info("Queries: {}".format(len(self.queries)))
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]
        ### encode corpus
        for corpus_start_idx in trange(0, len(self.corpus_ids), self.corpus_chunk_size, desc='Encode Corpus'):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus_ids))
            #sub_corpus_embeddings = torch.tensor(corpus_embeddings[corpus_start_idx:corpus_end_idx]).to(self.accelerator.device)
            batch_ids=self.corpus_ids[corpus_start_idx:corpus_end_idx]
            batch_texts=self.corpus[corpus_start_idx:corpus_end_idx]
            batch_embeddings = corpus_model.module.encode(
                batch_texts,
                show_progress_bar=self.accelerator.is_main_process, 
                batch_size=self.encode_batch_size, 
                convert_to_tensor=True, 
                num_proc=num_proc,
                accelerator=self.accelerator
            )
            ### save corpus embeddings
           # with open(self.corpus_embedding_path,'a+') as f:
           #     for batch_id,batch_embedding in zip(batch_ids,batch_embeddings):
            #        f.write(
             #           batch_id+'\t'+str(list(batch_embedding.cpu().numpy()))+'\n'
              #  )
            ### calculate chunk scores and save 
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, batch_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()
                with open(self.score_path,'a+') as f:
                    for query_itr in trange(0,len(query_embeddings),1,desc="save query-corpus score",disable=not self.accelerator.is_main_process):
                        #qid=self.queries_ids[query_itr]
                        for id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                            #queries_result_list[name][query_itr].append(
                            #    {   'corpus_id': corpus_ids[sub_corpus_id], 
                            #        'score': score
                            #    }
                            #    )
                            did=batch_ids[id]
                            qid=self.queries_ids[query_itr]
                            f.write(str(query_itr)+'\t'+qid+'\t'+did+'\t'+name+'\t'+str(score)+'\n')

        logger.info("Corpus: {}\n".format(len(self.corpus)))
            
        ### new iteration
        #tk = tqdm(
        #self.corpus_dataloader,
        #disable=not self.accelerator.is_local_main_process,
        #desc='Encode Corpus'
    #)
        ### encode corpus
        #for _,batch in enumerate(tk):
        #    batch_ids,batch_texts=batch['batch_ids'],batch['batch_texts']
        #    batch_embeddings = corpus_model.module.encode(
        #        batch_texts,
        #        show_progress_bar=False, 
        #        batch_size=self.encode_batch_size, 
        #        convert_to_tensor=True, 
        #        num_proc=num_proc,
        #        accelerator=self.accelerator
        #    )
        #    ### save corpus embeddings
        #    with open(self.corpus_embedding_path,'a+') as f:
        #        for batch_id,batch_embedding in zip(batch_ids,batch_embeddings):
        #            f.write(
        #                batch_id+'\t'+str(list(batch_embedding.cpu().numpy()))+'\n'
        #        )
        #Iterate over chunks of the corpus
        #for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
        #    corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
        #    if corpus_embeddings is None:
        #        sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True, num_proc=num_proc)
        #    else:
        #        sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute cosine similarites
        #    for name, score_function in self.score_functions.items():
        #        pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                #Get top-k values
        #        pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
        #        pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
        #        pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

        #        for query_itr in range(len(query_embeddings)):
        #            for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
        #                corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
        #                queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})
        if self.accelerator.is_local_main_process:
            ##### load scores and calculate metrics####

            #query_embeddings = model.module.encode(self.queries, show_progress_bar=True, batch_size=self.encode_batch_size, convert_to_tensor=True, num_proc=num_proc,accelerator=self.accelerator)
            
            #queries_result_list = {}
            #for name in self.score_functions:
            #    queries_result_list[name] = [[] for _ in range(len(query_embeddings))]
            
            ### load corpus embeddings        
            #corpus_ids=[]
            #corpus_embeddings=[]
            with open(self.score_path,'r') as f:
                for item in f:
                    pair_score=item.strip('\n').split('\t')
                    query_itr,did,name,score=eval(pair_score[0]),pair_score[2],pair_score[3],eval(pair_score[4])
                    queries_result_list[name][query_itr].append(
                                {   'corpus_id': did, 
                                    'score': score
                                }
                                )

            #with open(self.corpus_embedding_path,'r') as f:
            #    for item in tqdm(f,desc="Load Corpus Embeddings"):
            #        corpus=item.strip('\n').split('\t')
            #        id,embedding=corpus[0],eval(corpus[1])
            #       corpus_ids.append(id)
                    #corpus_embeddings.append(torch.tensor(embedding).to(self.accelerator.device))
            #        corpus_embeddings.append(embedding)
            
            #for corpus_start_idx in trange(0, len(corpus_ids), self.corpus_chunk_size, desc='Calculate Query-Corpus Score', disable=False):
            #    corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus_ids))
            #    sub_corpus_embeddings = torch.tensor(corpus_embeddings[corpus_start_idx:corpus_end_idx]).to(self.accelerator.device)

            #    for name, score_function in self.score_functions.items():
            #        pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                    #Get top-k values
            #        pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=True)
            #        pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
            #        pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

            #        for query_itr in range(len(query_embeddings)):
            #            for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
            #                queries_result_list[name][query_itr].append(
            #                    {   'corpus_id': corpus_ids[sub_corpus_id], 
            #                        'score': score
            #                    }
            #                    )
            #Compute scores
            scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

            #Output
            for name in self.score_function_names:
                logger.info("Score-Function: {}".format(name))
                self.output_scores(scores[name])
            # Write results to disc
            if self.results_save_folder is not None and self.write_csv:
                csv_path = os.path.join(self.results_save_folder, self.csv_file)
                if not os.path.isfile(csv_path):
                    fOut = open(csv_path, mode="w", encoding="utf-8")
                    fOut.write(",".join(self.csv_headers))
                    fOut.write("\n")

                else:
                    fOut = open(csv_path, mode="a", encoding="utf-8")

                output_data = [round,stage]
                for name in self.score_function_names:
                    #for k in self.accuracy_at_k:
                    #    output_data.append(scores[name]['accuracy@k'][k])

                    #for k in self.precision_recall_at_k:
                    #    output_data.append(scores[name]['precision@k'][k])
                    #    output_data.append(scores[name]['recall@k'][k])

                    for k in self.mrr_at_k:
                        output_data.append(scores[name]['mrr@k'][k])

                    for k in self.ndcg_at_k:
                        output_data.append(scores[name]['ndcg@k'][k])

                    #for k in self.map_at_k:
                    #    output_data.append(scores[name]['map@k'][k])

                fOut.write(",".join(map(str, output_data)))
                fOut.write("\n")
                fOut.close()
            #return scores
            #return scores['ndcg@k'][10]
            #create next stage's qrels_file_path:

        #return None

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        #precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        #recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        #AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            #top_hits=queries_result_list[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            #for k_val in self.accuracy_at_k:
            #    for hit in top_hits[0:k_val]:
            #        if hit['corpus_id'] in query_relevant_docs:
            #            num_hits_at_k[k_val] += 1
            #            break

            # Precision and Recall@k
            #for k_val in self.precision_recall_at_k:
            #    num_correct = 0
            #    for hit in top_hits[0:k_val]:
            #        if hit['corpus_id'] in query_relevant_docs:
            #            num_correct += 1

            #    precisions_at_k[k_val].append(num_correct / k_val)
            #    recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            #for k_val in self.map_at_k:
            #    num_correct = 0
            #    sum_precisions = 0

            #    for rank, hit in enumerate(top_hits[0:k_val]):
            #        if hit['corpus_id'] in query_relevant_docs:
            #            num_correct += 1
            #            sum_precisions += num_correct / (rank + 1)

            #    avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
            #    AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        #for k in num_hits_at_k:
        #    num_hits_at_k[k] /= len(self.queries)

        #for k in precisions_at_k:
        #    precisions_at_k[k] = np.mean(precisions_at_k[k])

        #for k in recall_at_k:
            #recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        #for k in AveP_at_k:
        #    AveP_at_k[k] = np.mean(AveP_at_k[k])

        return { 'ndcg@k': ndcg, 'mrr@k': MRR}
        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}


    def output_scores(self, scores):
        #for k in scores['accuracy@k']:
        #    logger.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

        #for k in scores['precision@k']:
        #    logger.info("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

        #for k in scores['recall@k']:
        #    logger.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))

        for k in scores['mrr@k']:
            logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

        for k in scores['ndcg@k']:
            logger.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))

        #for k in scores['map@k']:
        #    logger.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))


    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
