import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sgpt-125M", type=str)
parser.add_argument("--pooling", default="weightedmean", type=str)
parser.add_argument("--datasets", default="sgpt-125M", type=str)
parser.add_argument("--margin", default=0.1, type=int)
args = parser.parse_args()
## prepare qrels
qrels_dir="/data/home/scv0540/run/my_dr/datasets/{}/{}_{}-pooling_{}".format(args.datasets,args.model,args.pooling,args.datasets)
if os.path.exists(qrels_dir):
    os.system("rm -r {}".format(qrels_dir))
os.mkdir(qrels_dir)
qrels_path_list_file="{}/qrels_path.tsv".format(qrels_dir)
with open(qrels_path_list_file,'w') as f:
    pass
# prepare checkpoints
checkpoint_dir="/data/home/scv0540/run/my_dr/checkpoints/{}_{}-pooling_{}".format(args.model,args.pooling,args.datasets)
if os.path.exists(checkpoint_dir):
    os.system("rm -r {}".format(checkpoint_dir))
os.mkdir(checkpoint_dir)
checkpoint_path_list_file="{}/checkpoint_path.tsv".format(checkpoint_dir)
with open(checkpoint_path_list_file,'w') as f:
    pass
# prepare results files
results_dir="/data/home/scv0540/run/my_dr/results/{}_{}-pooling_{}".format(args.model,args.pooling,args.datasets)
if os.path.exists(results_dir):
    os.system( "rm -r {}".format(results_dir))
os.mkdir(results_dir)
# prepare score files
score_dir="/data/home/scv0540/run/my_dr/scores"
if not os.path.exists(score_dir):
    os.mkdir(score_dir)
# prepare logs files
logs_dir="/data/home/scv0540/run/my_dr/logs/{}_{}-pooling_{}".format(args.model,args.pooling,args.datasets)
if os.path.exists(logs_dir):
    os.system("rm -r {}".format(logs_dir))
os.mkdir(logs_dir)

