#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/9/20
#

import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import numpy as np
from train_util import *
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.stat_model_modified import DocReaderModel
#from src.bert_model_final import DocReaderModel
from src.batcher_statistics import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils import eval_bleu, eval_nist

##TODO ADDED
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm
from pynvml.smi import nvidia_smi
import statistics
from statistics import mean, median, stdev, pstdev, mode

nvsmi = nvidia_smi.getInstance()
def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    print( "%d %s" % (usage["used"], usage["unit"]))
##############

args = set_args()
# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger = create_logger(__name__, to_disk=True, log_file=args.log_file)


def main():
    opt = vars(args)
    #logger.info(opt) # TODO DELETED
    embedding, opt, vocab = load_meta(
        opt, args.meta)
    max_doc = opt['max_doc']
    max_query = opt['max_query']

    checkpoint_path = args.resume
    if checkpoint_path == '':
        if not args.if_train:
            print('checkpoint path can not be empty during testing...')
            exit()
        model = DocReaderModel(opt, embedding)  # TODO here is th model
    else:
        print("resume....", args.resume) # TODO ADDED
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model = DocReaderModel(opt, embedding, state_dict)
    model.setup_eval_embed(embedding)

    if args.cuda:
        model.cuda()


    toy_checkpoint = os.path.join(os.path.join(model_dir,opt["model_type"]))
    if not os.path.exists(toy_checkpoint):
        os.makedirs(toy_checkpoint)


    if args.if_train:
        train_time = datetime.now()
        logger.info('Loading training data')

        train_data = BatchGen(os.path.join(args.data_dir, args.train_data),
                              batch_size=args.batch_size,
                              gpu=args.cuda, doc_maxlen=max_doc,query_maxlen=max_query)

        logger.info('Loading dev data')
        dev_data = BatchGen(os.path.join(args.data_dir, args.dev_data),
                            batch_size=args.batch_size,
                            gpu=args.cuda, is_train=False, doc_maxlen=max_doc, query_maxlen=max_query)


        #TODO:MODIFIED
        for epoch in tqdm(range(0, 11)):#TODO args.epoches
            logger.warning('At epoch {}'.format(epoch))
            train_data.reset()
            start = datetime.now()
            #TODO:MODIFIED
            doc_len = []
            query_len = []

            for i, batch in enumerate(train_data):
              doc_len.extend(batch[1]['doc_len'])
              query_len.extend(batch[1]['query_len'])

            sns.distplot(doc_len, hist=True, kde=False,
             bins= 100, color = 'g',
             hist_kws={'edgecolor':'black'})

            #sns.distplot(doc_len, bins=10, color="g", ax=ax)
            plt.savefig(str(max_doc)+'word_doc.pdf')

            fig, ax = plt.subplots()
            sns.distplot(query_len, bins=50,  hist=True, kde=False, color="g", ax=ax)
            plt.savefig(str(max_doc)+'word_query.pdf')

            print("Mean doc length: ", mean(doc_len))
            print("stdev doc length: ", stdev(doc_len))
            print("pstdev doc length: ", pstdev(doc_len))
            print("median doc length: ", median(doc_len))
            print("max doc length: ", max(doc_len))
            print("mode doc length: ", mode(doc_len))

            print("max query len: ", max(query_len))
            print("Mean query len: ", mean(query_len))
            print("median query len: ", median(query_len))
            print("stdev query len: ", stdev(query_len))
            print("pstdev query len: ", pstdev(query_len))
            print("mode query length: ", mode(query_len))


            return






if __name__ == '__main__':
    main()
'''
##### full###
Mean doc length:  278.0371513168147                                                                 │12739 admin      20   0  106M  4964  3460 S  0.0  0.0  0:12.69 sshd: admin@pts/10
stdev doc length:  194.4375261818103                                                                │12851 admin      20   0 12988  2904  2472 S  0.0  0.0  0:13.32 watch -n 2 nvidia-smi
pstdev doc length:  194.43601115700358                                                              │  651 root       20   0 11212  1888  1748 S  0.0  0.0  0:11.88 /usr/bin/nvidia-persistenced --verbose
median doc length:  264.0                                                                           │12602 admin      20   0  105M  3684  2676 S  0.0  0.0  0:02.71 sshd: admin@pts/1
max doc length:  501                                                                                │  733 root       20   0 25988  3452  2176 S  0.0  0.0  0:09.74 /sbin/dhclient -1 -4 -v -pf /run/dhclie
mode doc length:  501                                                                               │  646 root       20   0  279M  6940  5980 S  0.0  0.0  0:04.41 /usr/lib/accountsservice/accounts-daemo
max query len:  101                                                                                 │  666 root       20   0  279M  6940  5980 S  0.0  0.0  0:04.34 /usr/lib/accountsservice/accounts-daemo
Mean query len:  31.650007791803024                                                                 │  399 root       20   0 46088  4740  3148 S  0.0  0.0  0:00.69 /lib/systemd/systemd-udevd
median query len:  24.0                                                                             │  657 root       20   0 30032  3136  2848 S  0.0  0.0  0:00.64 /usr/sbin/cron -f
stdev query len:  25.410542214058765                                                                │ 1409 admin      20   0 21372  5092  3416 S  0.0  0.0  0:00.02 -bash
pstdev query len:  25.41034421934775                                                                │    1 root       20   0  155M  9000  6760 S  0.0  0.0  0:03.66 /sbin/init maybe-ubiquity
mode query length:  101  

'''



'''

########## toy#############3


Mean doc length:  256.978                                                                           │12851 admin      20   0 12988  2904  2472 S  0.0  0.0  0:13.42 watch -n 2 nvidia-smi
stdev doc length:  209.5620720456833                                                                │  651 root       20   0 11212  1888  1748 S  0.0  0.0  0:11.94 /usr/bin/nvidia-persistenced --verbose
pstdev doc length:  209.3524050876894                                                               │12602 admin      20   0  105M  3684  2676 S  0.0  0.0  0:02.72 sshd: admin@pts/1
median doc length:  104.0                                                                           │    1 root       20   0  155M  9000  6760 S  0.0  0.0  0:03.67 /sbin/init maybe-ubiquity
max doc length:  501                                                                                │  733 root       20   0 25988  3452  2176 S  0.0  0.0  0:09.75 /sbin/dhclient -1 -4 -v -pf /run/dhclie
mode doc length:  501                                                                               │  646 root       20   0  279M  6940  5980 S  0.0  0.0  0:04.42 /usr/lib/accountsservice/accounts-daemo
max query len:  101                                                                                 │  666 root       20   0  279M  6940  5980 S  0.0  0.0  0:04.35 /usr/lib/accountsservice/accounts-daemo
Mean query len:  26.76                                                                              │  399 root       20   0 46088  4740  3148 S  0.0  0.0  0:00.69 /lib/systemd/systemd-udevd
median query len:  19.0                                                                             │  657 root       20   0 30032  3136  2848 S  0.0  0.0  0:00.64 /usr/sbin/cron -f
stdev query len:  22.763540091888828                                                                │ 1409 admin      20   0 21372  5092  3416 S  0.0  0.0  0:00.02 -bash
pstdev query len:  22.740765158630875                                                               │  540 systemd-n  20   0 71892  5304  4704 S  0.0  0.0  0:00.39 /lib/systemd/systemd-networkd
mode query length:  10 
'''