import os
import time
import argparse
import random
from sklearn import datasets
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sgpt-125M", type=str)
parser.add_argument("--pooling", default="weightedmean", type=str)
parser.add_argument("--datasets", default="sgpt-125M", type=str)
parser.add_argument("--margin", default=0.1, type=int)
args = parser.parse_args()

dir="/data/home/scv0540/run/my_dr/checkpoints/{}_{}-pooling_{}".format(args.model,args.pooling,args.datasets)
if os.path.exists(dir):
    os.system("rm -r {}".format(dir))
os.mkdir(dir)
checpoint_path_list_file="{}/checkpoint_path.tsv".format(dir)
with open(checpoint_path_list_file,'w') as f:
    pass
checkpoint_path_list=[]
times=1
while True:
    with open(checpoint_path_list_file,'r') as f:
        not_find=True
        for item in f:
            round_num,stage_num,path=item.strip('\n').split('\t')
            round_num,stage_num=int(round_num),int(stage_num)
            if (round_num,stage_num) not in checkpoint_path_list:
                checkpoint_path_list.append((round_num,stage_num) )
                not_find=False
                print("******check-{} find new checkpoint in {}******\n".format(times,path))
                os.system("srun --job-name=inference_{}_{}-pooling_{}_round{}-stage{}_test --nodes=1 --gpus=8 --mem=300G --exclusive bash inference.sh {} {} {} {} {} {}".format(args.model,args.pooling,args.datasets,round_num,stage_num,args.model,args.pooling,args.datasets,round_num,stage_num,"test") )   
        if not_find:
            print("******the {}'s check don't find new checkpoint******\n".format(times))
        time.sleep(600)
        times+=1
