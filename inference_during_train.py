import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sgpt-125M", type=str)
parser.add_argument("--pooling", default="weightedmean", type=str)
parser.add_argument("--datasets", default="sgpt-125M", type=str)
args = parser.parse_args()

dir="/data/home/scv0540/run/my_dr/checkpoints/{}_{}-pooling_{}".format(args.model,args.pooling,args.datasets)
map_file="{}/checkpoint_path.tsv".format(dir)
epoch_list=[]
times=1
while True:
    with open(map_file,'r') as f:
        not_find=True
        for item in f:
            epoch,path=item.strip('\n').split('\t')
            epoch_num=epoch.split('_')[1]
            path=path.split(":")[1]
            if epoch_num not in epoch_list:
                not_find=False
                print("******check-{} find new checkpoint in {}******\n".format(times,path))
                os.system("srun --job-name=inference_{}_{}-pooling_{}_epoch{} --nodes=1 --gpus=8 --mem=300G --exclusive bash inference.sh {} {} {} {}".format(args.model,args.pooling,args.datasets,epoch_num,args.model,args.pooling,args.datasets,epoch_num) )
                epoch_list.append(epoch_num)

        if not_find:
            print("******the {}'s check don't find new checkpoint******\n".format(times))
        time.sleep(600)
        times+=1
