import os
import time
import argparse
import random
from tqdm import tqdm
from tqdm import trange
models=[
    "released_gpt125M"
]
beir_datasets=[
         'scidocs',
          'scifact',
           'fiqa',
            'nfcorpus'
 ]
for model in models:
    for beir_dataset in beir_datasets:
        results_dir="/data/home/scv0540/run/my_dr/results/{}/{}".format(beir_dataset,model)
        if os.path.exists(results_dir):
            os.system( "rm -r {}".format(results_dir))
        os.system("mkdir -p {}".format(results_dir))
        # prepare score files
        score_dir="/data/home/scv0540/run/my_dr/scores/{}".format(beir_dataset)
        if not os.path.exists(score_dir):
            os.system("mkdir -p {}".format(score_dir))
        os.system("srun --job-name=beir_inference_release --nodes=1 --gpus=8 --mem=300G --exclusive bash inference_beir_release.sh {}".format(beir_dataset))
