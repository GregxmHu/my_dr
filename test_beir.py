import os
import time
import argparse
import random
from tqdm import tqdm
from tqdm import trange
models=[
    ('with-hard_t5-base-bootstrap-fulltune-totaltound1-epoch-5','cls',0,0)
]
beir_datasets=['nfcorpus',
         'scifact',
          'scidocs',
           'fiqa','arguana'
 ]
for model,pooling,round_idx,stage_idx in models:
    for beir_dataset in beir_datasets:
        results_dir="/data/home/scv0540/run/my_dr/results/{}/{}_{}-pooling_msmarco".format(beir_dataset,model,pooling)
        if os.path.exists(results_dir):
            os.system( "rm -r {}".format(results_dir))
        os.system("mkdir -p {}".format(results_dir))
        # prepare score files
        score_dir="/data/home/scv0540/run/my_dr/scores/{}".format(beir_dataset)
        if not os.path.exists(score_dir):
            os.system("mkdir -p {}".format(score_dir))
        os.system("srun --job-name=beir_inference_{}_{}-pooling_{}_round{}-stage{} --nodes=1 --gpus=8 --mem=300G --exclusive bash inference_v2_beir.sh {} {} {} {} {} {}".format(model,pooling,'msmarco',round_idx,stage_idx,model,pooling,'msmarco',round_idx,stage_idx,beir_dataset))
