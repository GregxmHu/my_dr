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

logger = logging.getLogger(__name__)

class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 train_queries: Dict[str, str],  #qid => query
                 train_relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 test_queries: Dict[str, str],  #qid => query
                 test_relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus: Dict[str, str],  #cid => doc
                 corpus_chunk_size: int = 320,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [10],
                 precision_recall_at_k: List[int] = [10],
                 map_at_k: List[int] = [10],
                 show_progress_bar: bool = False,
                 encode_batch_size: int = 32,
                 #name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim},       #Score function, higher=more similar
                 main_score_function: str = None,
                 accelerator: Accelerator = None,
                 #corpus_embedding_path: str="/data/private/huxiaomeng/sgpt/corpus_embeddings/msmarco_small.txt",
                 results_save_folder: str=None,
                 train_score_path: str="/data/private/huxiaomeng/sgpt/corpus_embeddings/sgpt-125M-scifact_topk_score.txt",
                 test_score_path:str="/data/private/huxiaomeng/sgpt/corpus_embeddings/sgpt-125M-scifact_topk_score.txt"
                 ):
        self.train_score_path=train_score_path
        self.test_score_path=test_score_path
        self.corpus_chunk_size=corpus_chunk_size
        self.encode_batch_size=encode_batch_size
        #self.corpus_embedding_path=corpus_embedding_path
        self.results_save_folder=results_save_folder
        self.train_queries_ids = []
        self.test_queries_ids = []
        for qid in train_queries:
            if qid in train_relevant_docs and len(train_relevant_docs[qid]) > 0:
                self.train_queries_ids.append(qid)

        self.train_queries = [train_queries[qid] for qid in self.train_queries_ids]
        
        for qid in test_queries:
            if qid in test_relevant_docs and len(test_relevant_docs[qid]) > 0:
                self.test_queries_ids.append(qid)

        self.test_queries = [test_queries[qid] for qid in self.test_queries_ids]
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
        self.train_relevant_docs = train_relevant_docs
        self.test_relevant_docs = test_relevant_docs
        #self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.encode_batch_size = encode_batch_size
        #self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        #if name:
        #    name = "_" + name

        self.train_csv_file: str = "train_results.csv"
        self.test_csv_file: str = "test_results.csv"
        self.train_csv_headers = ["round","stage"]
        self.test_csv_headers = ["round","stage"]

        for score_name in self.score_function_names:
            #for k in accuracy_at_k:
            #    self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            #for k in precision_recall_at_k:
            #    self.csv_headers.append("{}-Precision@{}".format(score_name, k))
            #    self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.train_csv_headers.append("{}-MRR@{}".format(score_name, k))
                self.test_csv_headers.append("{}-MRR@{}".format(score_name, k))
            for k in ndcg_at_k:
                self.train_csv_headers.append("{}-NDCG@{}".format(score_name, k))
                self.test_csv_headers.append("{}-NDCG@{}".format(score_name, k))
            #for k in map_at_k:
            #    self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model,  round: int = 1,stage: int =1,  num_proc: int = None, *args, **kwargs) -> float:
        logger.info("Information Retrieval Evaluation and Refresh negatives" + " after round {} stage {}".format(round,stage))

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
        train_query_embeddings = model.module.encode(self.train_queries, show_progress_bar=self.accelerator.is_main_process, batch_size=self.encode_batch_size, convert_to_tensor=True, num_proc=num_proc,accelerator=self.accelerator)
        test_query_embeddings = model.module.encode(self.test_queries, show_progress_bar=self.accelerator.is_main_process, batch_size=self.encode_batch_size, convert_to_tensor=True, num_proc=num_proc,accelerator=self.accelerator)

        train_queries_result_list = {}
        test_queries_result_list = {}
        logger.info("Train Queries: {}   Test Queries: {}".format(len(self.train_queries),len(self.test_queries)))
        for name in self.score_functions:
            #train_queries_result_list[name] = [[] for _ in range(len(train_query_embeddings))]
            test_queries_result_list[name] = [[] for _ in range(len(test_query_embeddings))]
        ### encode corpus
        for corpus_start_idx in trange(0, len(self.corpus_ids), self.corpus_chunk_size, desc='Encode Corpus',disable=not self.accelerator.is_main_process):
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
            for train_query_start_idx in trange(0,len(self.train_queries_ids),2500,disable=True):
                train_query_end_idx = min(train_query_start_idx + 2500, len(self.train_queries_ids))
                train_sub_query_embeddings=train_query_embeddings[train_query_start_idx:train_query_end_idx]
                for name, score_function in self.score_functions.items():
                    train_pair_scores = score_function(train_sub_query_embeddings, batch_embeddings)

                    #Get top-k values
                    train_pair_scores_top_k_values, train_pair_scores_top_k_idx = torch.topk(train_pair_scores, 1, dim=1, largest=True, sorted=False)
                    train_pair_scores_top_k_values = train_pair_scores_top_k_values.cpu().tolist()
                    train_pair_scores_top_k_idx = train_pair_scores_top_k_idx.cpu().tolist()
                    with open(self.train_score_path,'a+') as f:
                        for train_sub_query_itr in trange(0,len(train_sub_query_embeddings),1,disable=True):
                            #qid=self.queries_ids[query_itr]
                            for id, train_score in zip(train_pair_scores_top_k_idx[train_sub_query_itr], train_pair_scores_top_k_values[train_sub_query_itr]):
                                #queries_result_list[name][query_itr].append(
                                #    {   'corpus_id': corpus_ids[sub_corpus_id], 
                                #        'score': score
                                #    }
                                #    )
                                train_query_itr=train_query_start_idx+train_sub_query_itr
                                did=batch_ids[id]
                                train_qid=self.train_queries_ids[train_query_itr]
                                f.write(str(train_query_itr)+'\t'+train_qid+'\t'+did+'\t'+name+'\t'+str(train_score)+'\n')
                   # if self.accelerator.is_main_process:
                   #     print("save train query-passage scores")
            for name, score_function in self.score_functions.items():
                test_pair_scores = score_function(test_query_embeddings, batch_embeddings)

                #Get top-k values
                test_pair_scores_top_k_values, test_pair_scores_top_k_idx = torch.topk(test_pair_scores, min(max_k, len(test_pair_scores[0])), dim=1, largest=True, sorted=False)
                test_pair_scores_top_k_values = test_pair_scores_top_k_values.cpu().tolist()
                test_pair_scores_top_k_idx = test_pair_scores_top_k_idx.cpu().tolist()
                with open(self.test_score_path,'a+') as f:
                    for test_query_itr in trange(0,len(test_query_embeddings),1,disable=True):
                        #qid=self.queries_ids[query_itr]
                        for id, test_score in zip(test_pair_scores_top_k_idx[test_query_itr], test_pair_scores_top_k_values[test_query_itr]):
                            #queries_result_list[name][query_itr].append(
                            #    {   'corpus_id': corpus_ids[sub_corpus_id], 
                            #        'score': score
                            #    }
                            #    )
                            #train_query_itr=train_query_start_idx+train_sub_query_itr
                            did=batch_ids[id]
                            test_qid=self.test_queries_ids[test_query_itr]
                            f.write(str(test_query_itr)+'\t'+test_qid+'\t'+did+'\t'+name+'\t'+str(test_score)+'\n')
            if self.accelerator.is_main_process:
                print("save test query-passage scores")
        logger.info("Corpus: {}\n".format(len(self.corpus)))
            
        
        if self.accelerator.is_local_main_process:
            
            

            with open(self.test_score_path,'r') as f:
                for item in f:
                    pair_score=item.strip('\n').split('\t')
                    query_itr,did,name,score=eval(pair_score[0]),pair_score[2],pair_score[3],eval(pair_score[4])
                    test_queries_result_list[name][query_itr].append(
                                {   'corpus_id': did, 
                                    'score': score
                                }
                                )

            test_scores = {name: self.compute_metrics(test_queries_result_list[name],'test') for name in self.score_functions}
            #Output
            for name in self.score_function_names:
                logger.info("Score-Function: {}".format(name))
                logger.info("On test set")
                self.output_scores(test_scores[name])
            # Write train results 
            
            #return scores
            # Write train results 
            if self.results_save_folder is not None and self.write_csv:
                test_csv_path = os.path.join(self.results_save_folder, self.test_csv_file)
                if not os.path.isfile(test_csv_path):
                    fOut = open(test_csv_path, mode="w", encoding="utf-8")
                    fOut.write(",".join(self.test_csv_headers))
                    fOut.write("\n")

                else:
                    fOut = open(test_csv_path, mode="a", encoding="utf-8")

                output_data = [round,stage]
                for name in self.score_function_names:
                    #for k in self.accuracy_at_k:
                    #    output_data.append(scores[name]['accuracy@k'][k])

                    #for k in self.precision_recall_at_k:
                    #    output_data.append(scores[name]['precision@k'][k])
                    #    output_data.append(scores[name]['recall@k'][k])

                    for k in self.mrr_at_k:
                        output_data.append(test_scores[name]['mrr@k'][k])

                    for k in self.ndcg_at_k:
                        output_data.append(test_scores[name]['ndcg@k'][k])

                    #for k in self.map_at_k:
                    #    output_data.append(scores[name]['map@k'][k])

                fOut.write(",".join(map(str, output_data)))
                fOut.write("\n")
                fOut.close()
            #return scores['ndcg@k'][10]
            #create next stage's qrels_file_path:

        #return None

    def compute_metrics(self, queries_result_list: List[object],mode: str):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        #precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        #recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        #AveP_at_k = {k: [] for k in self.map_at_k}
        if mode =="train":
            queries_ids=self.train_queries_ids
            relevant_docs=self.train_relevant_docs
            queries=self.train_queries
        else:
            queries_ids=self.test_queries_ids
            relevant_docs=self.test_relevant_docs
            queries=self.test_queries
        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            #top_hits=queries_result_list[query_itr]
            query_relevant_docs = relevant_docs[query_id]

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
            MRR[k] /= len(queries)

        #for k in AveP_at_k:
        #    AveP_at_k[k] = np.mean(AveP_at_k[k])

        return { 'ndcg@k': ndcg, 'mrr@k': MRR}
        #return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}


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


