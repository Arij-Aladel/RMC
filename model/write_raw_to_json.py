import os
import json
import pickle
from functools import partial
from collections import Counter
from my_utils.tokenizer import Vocabulary, reform_text
from my_utils.word2vec_utils import load_glove_vocab, build_embedding  # edited install numpy and torch
# from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from config import set_args
from process_raw_data import combine_files as load_reddit_data
from process_raw_data import write_files
max_vocab_size = 30000
#from io import open  # TODO added to try fix line 155



def load_vocab(path):
    vocab = []
    with open(path) as fr:
        for line in fr:
            vocab.append(line.strip())
    vocab = Vocabulary.build(vocab)
    logger.info('final vocab size: {}'.format(len(vocab)))
    return vocab

def build_vocab(data):
    logger.info('Collect vocab')
    query_answer_counter = Counter()
    doc_counter = Counter()
    vocab = []
    for sample in data:
        query_answer_counter.update(sample['query'] + sample['response'])
        doc_counter.update(sample['fact'])

    counter = query_answer_counter + doc_counter
    print("type counter: ", type(counter))
    tmp_vocab = [w for w in query_answer_counter]
    tmp_vocab += [w for w in doc_counter.keys() - query_answer_counter.keys()]  # TODO: fix it from doc_counter.keys() - query_answer_counter.keys() to doc_counter - query_answer_counter
    tmp1 = list(set([w for w in counter]))
    import collections # TODO: delete it!
    print("My idea is right!!!!: ", collections.Counter(tmp_vocab ) == collections.Counter(tmp1))
    remove_vocab = []
    logger.info('finding www words ...')
    for x in tmp_vocab:
        if 'http' in x:
            remove_vocab.append(x)
    logger.info('finding www words done!')     

    for x in tmp_vocab:
        if x not in remove_vocab:
            vocab.append(x)

    logger.info('sorting ...')  
    vocab = sorted(vocab, key=counter.get, reverse=True)
    # truncate
    vocab = vocab[:max_vocab_size]
    vocabulary = Vocabulary()# the fix is here
    vocab = Vocabulary.build(vocab)
    logger.info('final vocab size: {}'.format(len(vocab)))
    with open('./vocab.txt', 'w') as fin:
        for x in vocab:
            fin.write(x + '\n')
    return vocab

def build_data(data, vocab, fout):
    with open(fout, 'w', encoding='utf-8') as writer:
        dropped_sample = 0
        for sample in data:
            fd = feature_func(sample, vocab)
            if fd is None:
                dropped_sample += 1
                continue
            writer.write('{}\n'.format(json.dumps(fd)))
        logger.info('dropped {} in total {}'.format(dropped_sample, len(data)))

def feature_func(sample, vocab):
    query_tokend = sample['query']
    doc_tokend = sample['fact']
    answer_tokend = sample['response']

    # features
    fea_dict = {}

    fea_dict['uid'] = sample['conv_id']
    fea_dict['hash_id'] = sample['hash_id']

    # TODO
    fea_dict['query_tok'] = tok_func(query_tokend, vocab)

    fea_dict['query_pos'] = []
    fea_dict['query_ner'] = []
  
    # TODO
    fea_dict['doc_tok'] = tok_func(doc_tokend, vocab)

    fea_dict['doc_pos'] = []
    fea_dict['doc_ner'] = []
    fea_dict['doc_fea'] = ''

    # TODO
    fea_dict['answer_tok'] = tok_func(answer_tokend, vocab)


    if len(fea_dict['query_tok']) == 0:
        fea_dict['query_tok'] = [0]
    if len(fea_dict['doc_tok']) == 0:
        fea_dict['doc_tok'] = [0]
    if len(fea_dict['answer_tok']) ==0:
        fea_dict['answer_tok'] = [0]
                
    return fea_dict

def tok_func(toks, vocab):
    return [vocab[w] for w in toks]

def main():
    args = set_args()
    global logger
    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)
    logger.info('Processing dataset')
    #TODO raw_data folder added to model folder
    train_path = os.path.join(args.raw_data_dir, 'train') # .json added
    valid_path = os.path.join(args.raw_data_dir, 'dev')   # .json added
    test_path = os.path.join(args.raw_data_dir, 'test')   # .json added
    logger.info('The path of training data: {}'.format(train_path))
    logger.info('The path of validation data: {}'.format(valid_path))
    logger.info('The path of test data: {}'.format(test_path))
    logger.info('{}-dim word vector path: {}'.format(args.glove_dim, args.glove))
    glove_path = args.glove
    glove_dim = args.glove_dim
    # set_environment(args.seed)
# TODO copy train, dev , test to data folder
    # load data
    train_data = load_reddit_data(train_path, anc_type='section',
	fact_len = 12, just_anc = False, is_train = True)
    valid_data = load_reddit_data(valid_path,anc_type='section',
	fact_len = 12, just_anc = False, is_train = False)
    test_data = load_reddit_data(test_path, anc_type='section',
    fact_len=12, just_anc = False, is_train=False)
    logger.info('#train data: {}'.format(len(train_data)))
    logger.info('#valid data: {}'.format(len(valid_data)))
    logger.info('#test data: {}'.format(len(test_data)))
    meta_path = args.meta
    print("meta_path::::", meta_path)
    if not os.path.exists(meta_path):
        logger.info('Build vocabulary')
        vocab = build_vocab(train_data + valid_data)
        logger.info('building embedding')
        embedding = build_embedding(glove_path, vocab, glove_dim)
        logger.info('emb done')
        meta = {'vocab': vocab, 'embedding': embedding}
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
    else:
        with open(meta_path, 'rb') as f:# TODO rb->r
            meta = pickle.load(f)  # TODO: fixed fix_imports=True, encoding = 'latin-1' doesnot help , pickle instead of pick solved the problem is added
            print("loaded meta:", meta.keys())
            vocab = meta['vocab']

    train_fout = os.path.join(args.data_dir, args.train_data)
    build_data(train_data, vocab, train_fout)
    logger.info('train data done')

    dev_fout = os.path.join(args.data_dir, args.dev_data)
    build_data(valid_data, vocab, dev_fout)
    logger.info('valid data done')

    test_fout = os.path.join(args.data_dir, args.test_data)
    build_data(test_data, vocab, test_fout)
    logger.info('test data done')
    logger.info('start writing processed data to train folder')
    write_files(args.data_dir + '/train', train_data)
    logger.info('start writing processed data to dev folder')
    write_files(args.data_dir + '/dev', valid_data)
    logger.info('start writing processed data to test folder')
    write_files(args.data_dir + '/test', test_data)

if __name__ == '__main__':
    main()

