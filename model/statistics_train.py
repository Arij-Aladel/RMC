#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/9/20
#


#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/9/20
#

import re
import os
import shutil
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import numpy as np
from stat_train_util import *
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.stat_model_modified import DocReaderModel
# from src.bert_model_final import DocReaderModel
from src.batcher_statistics import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils import eval_bleu, eval_nist

##TODO ADDED
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
from tqdm import tqdm
from pynvml.smi import nvidia_smi

nvsmi = nvidia_smi.getInstance()


def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    print("%d %s" % (usage["used"], usage["unit"]))


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
    # logger.info(opt) # TODO DELETED
    embedding, opt, vocab = load_meta(
        opt, args.meta)
    max_doc = opt['max_doc']
    max_query = opt['max_query']
    print("max_query: ", max_query)
    smooth = opt['smooth']
    is_rep = opt['is_rep']
    eval_step = opt['eval_step']  # 50
    curve_file = opt['curve_file']
    training_step = 0
    cur_eval_step = 1

    checkpoint_path = args.resume
    if checkpoint_path == '':
        if not args.if_train:
            exit()
        model = DocReaderModel(opt, embedding)  # TODO here is th model
    else:
        print("resume....", args.resume)  # TODO ADDED
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model = DocReaderModel(opt, embedding, state_dict)
    model.setup_eval_embed(embedding)
    logger.info("Total number of params: {}".format(model.total_param))

    if args.cuda:
        model.cuda()

    toy_checkpoint = os.path.join(os.path.join(model_dir, opt["model_type"]))
    if not os.path.exists(toy_checkpoint):
        os.makedirs(toy_checkpoint)

    def _eval_output(file_path=args.dev_data, full_path=args.dev_full, test_type='dev'):
        data = BatchGen(os.path.join(args.data_dir, file_path),
                        batch_size=args.batch_size,
                        gpu=args.cuda, is_train=False)

        full_path = os.path.join(args.data_dir, full_path)
        # print("test type: ", test_type)

        pred_output_path = os.path.join(toy_checkpoint, 'pred_toy/' + test_type) + "/"
        if not os.path.exists(pred_output_path):
            os.makedirs(pred_output_path)
        full_output_path = os.path.join(toy_checkpoint, 'full_toy/' + test_type) + "/"
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)

        t = args.test_output

        pred_output = pred_output_path + t + '.txt'
        full_output = full_output_path + t + '_full.txt'

        bleu, bleu_fact, diver_uni, diver_bi = \
            check(model,
                  data,
                  vocab,
                  full_path,
                  pred_output,
                  full_output, )
        _loss = eval_test_loss(model, data, vocab)
        _loss = _loss.data.cpu().numpy()
        logger.info('dev loss[{0:.5f}] ppl[{1:.5f}]'.format(
            _loss,
            np.exp(_loss)))
        print('DocReader Updates: {0}, Average Train Loss: {1:.5f}, e(Average Train Loss): {2:.5f},'
              ' Eval Loss: {3:.5f}, e(Eval Loss): {4:.5f}, BLeU: {5:.5f}, '
              'Diversity (Uni): {6:.5f}, Diversity(Bi): {7:.5f}, BleU(fact): {8:.5f}\n'.format(
            model.updates, model.train_loss.avg, np.exp(model.train_loss.avg), _loss, np.exp(_loss),
            float(bleu), float(diver_uni), float(diver_bi), float(bleu_fact)))

    ## end function

    if args.if_train:
        train_time = datetime.now()
        logger.info('Loading training data')

        train_data = BatchGen(os.path.join(args.data_dir, args.train_data),
                              batch_size=args.batch_size,
                              gpu=args.cuda, doc_maxlen=max_doc, query_maxlen=max_query)

        logger.info('Loading dev data')
        dev_data = BatchGen(os.path.join(args.data_dir, args.dev_data),
                            batch_size=args.batch_size,
                            gpu=args.cuda, is_train=False, doc_maxlen=max_doc, query_maxlen=max_query)

        curve_file = os.path.join(toy_checkpoint, curve_file)
        with open(curve_file, 'a+') as fout_dev:
            fout_dev.write('{0},{1},{2},{3},{4},'
                           '{5},{6},{7},{8},{9},{10}\n'.format("epoch", "batch",
                                                               "model.updates", "model.train_loss.avg",
                                                               "np.exp(model.train_loss.avg)", "dev_loss",
                                                               "np.exp(dev_loss)",
                                                               "float(bleu)", "float(diver_uni)", "float(diver_bi)",
                                                               "float(bleu_fact)"))
        full_path = os.path.join(args.data_dir, args.dev_full)  # full path od validation data dev ot test

        from statistics import mean, median, stdev, pstdev
        # TODO:MODIFIED
        for epoch in tqdm(range(0, 11)):  # TODO args.epoches
            start_epoch = datetime.now()
            logger.warning('At epoch {}'.format(epoch))
            train_data.reset()
            start = datetime.now()
            # TODO:MODIFIED
            # doc_len = []
            # query_len = []
            loss_doc = 0  # number of
            loss_query = 0
            total_num = 0

            for i, batch in enumerate(train_data):
                #   doc_len.append(batch[1]['doc_len'])
                #   query_len.append(batch[1]['query_len'])

                # #Plot Data
                # fig, ax = plt.subplots()
                # sns.distplot(doc_len, bins=22, color="g", ax=ax)
                # plt.savefig('501word_doc.pdf')

                # fig, ax = plt.subplots()
                # sns.distplot(query_len, bins=22, color="g", ax=ax)
                # plt.savefig('2501word_query.pdf')

                # print("Average doc length: ", mean(doc_len))
                # print("Mean query len: ", mean(query_len))

                # print("stdev doc length: ", stdev(doc_len))
                # print("stdev query len: ", stdev(query_len))

                # print("pstdev doc length: ", pstdev(doc_len))
                # print("pstdev query len: ", pstdev(query_len))

                # print("median doc length: ", median(doc_len))
                # print("median query len: ", median(query_len))

                # print("max doc length: ", max(doc_len))
                # print("max query len: ", max(query_len))

                # return
                training_step += 1
                checkstart = datetime.now()  # TODO ADDED
                _loss_doc, _loss_query = model.update(batch[0], vocab, smooth, is_rep)
                loss_doc += _loss_doc
                loss_query += _loss_query
                total_num += len(batch[0]["doc_tok"])

            #     # num_doc =
            #     # num_query =
            #     # print("\nUpdate time",str((datetime.now() - checkstart) /60), " minutes")  # TODO ADDED

            #     # return
            #     if (i + 1) % args.log_per_updates == 0:  # args.log_per_updates =
            #         logger.info('updates[{0:6}] train: loss[{1:.5f}]'
            #                     ' ppl[{2:.5f}] remaining[{3}]'.format(
            #             model.updates,
            #             model.train_loss.avg,
            #             np.exp(model.train_loss.avg),
            #             str((datetime.now() - start) / (i + 1) * (len(train_data) - i - 1)).split('.')[0]))

            #         # setting up scheduler
            #         if model.scheduler is not None:
            #             if opt['scheduler_type'] == 'rop':
            #                 model.scheduler.step(model.train_loss.avg, epoch=epoch)
            #             else:
            #                 model.scheduler.step()

            #     dev_loss = 0.0
            #     if (training_step) == cur_eval_step:  # TODO: need to understand it
            #         print("(training_step) == cur_eval_step", cur_eval_step)  # TODO ADDED
            #         print('evaluating_step is {} ....'.format(training_step))

            #         checkstart = datetime.now()  # TODO ADDED

            #         pred_output_path = os.path.join(toy_checkpoint, 'pred_toy/dev') + "/"
            #         if not os.path.exists(pred_output_path):
            #             os.makedirs(pred_output_path)

            #         full_output_path = os.path.join(toy_checkpoint, 'full_toy/dev') + "/"
            #         if not os.path.exists(full_output_path):
            #             os.makedirs(full_output_path)

            #         pred_output = os.path.join(pred_output_path, str(model.updates)) + '.txt'
            #         full_output = os.path.join(full_output_path, str(model.updates)) + '_full.txt'

            #         bleu, bleu_fact, diver_uni, diver_bi = check(model,
            #                                                      dev_data,
            #                                                      vocab,
            #                                                      full_path,  # full path od validation data dev ot test
            #                                                      pred_output,
            #                                                      full_output, )  # TODO understand why do we need to check and check for what?

            #         print("\ncheck time", str((datetime.now() - checkstart) / 60), " minutes")  # TODO ADDED
            #         checkstart = datetime.now()
            #         dev_loss = eval_test_loss(model, dev_data, vocab)  # TODO: the problem was here solved
            #         dev_loss = dev_loss.data.cpu().numpy()
            #         print("time evaluation: ", datetime.now() - checkstart, ", eval_test_loss done")  # TODO ADDED
            #         logger.info('\nupdates[{0:6}] train: loss[{1:.5f}] exp(train)[{2:.5f}]'
            #                     'dev: loss[{3:.5f}] exp(dev)[{4:.5f}]'.format(
            #             model.updates,
            #             model.train_loss.avg,
            #             np.exp(model.train_loss.avg),
            #             dev_loss,
            #             np.exp(dev_loss)))

            #         with open(curve_file, 'a+') as fout_dev:
            #             fout_dev.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f},{6:.5f},'
            #                            '{7:.5f},{8:.5f},{9:.5f},{10:.5f}\n'.format(epoch, i,
            #                                                                        model.updates, model.train_loss.avg,
            #                                                                        np.exp(model.train_loss.avg),
            #                                                                        dev_loss, np.exp(dev_loss),
            #                                                                        float(bleu), float(diver_uni),
            #                                                                        float(diver_bi), float(
            #                     bleu_fact)))  # TODO why np.exp?

            #         if cur_eval_step == 1:  # TODO: need to understand it
            #             print("eval_step: ", eval_step, "cur_eval_step", cur_eval_step)
            #             cur_eval_step = cur_eval_step - 1
            #         cur_eval_step += eval_step

            # # save
            # dev_loss = eval_test_loss(model, dev_data, vocab)
            # dev_loss = dev_loss.data.cpu().numpy()
            # if epoch % 10 == 0:
            #     logger.info('Saved model as checkpoint_epoch_{0}_{1}_{2:.5f}.pt'
            #                 .format(epoch, args.learning_rate, np.exp(dev_loss)))
            #     model_file = os.path.join(toy_checkpoint, 'checkpoint_epoch_{0}_{1}_{2:.5f}.pt'
            #                               .format(epoch, args.learning_rate, np.exp(dev_loss)))
            #     model.save(model_file, epoch)

            logger.info("Epoch {} time: {}".format(epoch, datetime.now() - start_epoch))
            logger.info("Num truncated documents{} of total documents {} , {} %".format(loss_doc, total_num,
                                                                                        loss_doc * 100 / total_num))
            logger.info("Num truncated queries{} of total queries {} , {} %".format(loss_query, total_num,
                                                                                    loss_query * 100 / total_num))
            return

        # TODO: ADDED
        logger.info("Train time: {}".format(datetime.now() - train_time))

        if args.model_type == 'BERT':
            BERT_after = "checkpoint/BERT/dev_curve.csv"
            BERT_after_df = pd.read_csv(BERT_after)
            BERT_after_df.plot(x="epoch", y=["model.train_loss.avg", "dev_loss"], title="BERT loss")
            plt.savefig('BERT_loss.pdf')

        if args.model_type == 'san':
            SAN = "checkpoint/san/dev_curve.csv"
            SAN_df = pd.read_csv(SAN)
            SAN_df.plot(x="epoch", y=["model.train_loss.avg", "dev_loss"], title='SAN loss')
            plt.savefig('SAN_loss.pdf')



    else:
        test_time = datetime.now()
        logger.info('Loading evaluation data')
        checkpoint_path = args.resume
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model = DocReaderModel(opt, embedding, state_dict)
        model.setup_eval_embed(embedding)
        logger.info("Total number of params: {}".format(model.total_param))
        if args.cuda:
            model.cuda()

        print('test result is:')
        _eval_output(args.test_data, args.test_full, 'test')
        logger.info("Test time: {}".format(datetime.now() - test_time))  # TODO ADDED
        # TODO: ADDED
        if args.model_type == 'BERT':
            shutil.copy('checkpoint/toy104d_24q_bert205/BERT/full_toy/test/submission_full.txt',
                        '../evaluation/dstc/BERT.txt')
        if args.model_type == 'san':
            shutil.copy('checkpoint/toy104d_24q_bert205/san/full_toy/test/submission_full.txt',
                        '../evaluation/dstc/SAN.txt')


if __name__ == '__main__':
    main()
