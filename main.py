# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
# os.environ['CUDA_VISIBLE_DEVICES']=''
import time
import csv
import multiprocessing as mp
from param import *
from utils import *
from sim_env.env import Environment

def main():
    # set up environment
    env = Environment()
    # env.reset()
    end = 0
    begin = 0
    env.loadData()
    done = env.step(args.start_node, args.end_node, args.begin_index, args.end_index,args.total_time,args.route_alg,args.process_alg)


if __name__ == '__main__':
    main()
