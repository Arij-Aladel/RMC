#!/bin/bash

if [ ! -d "log" ]; then
  mkdir log
fi
CUDA_VISIBLE_DEVICES=1 \
python3  train_stat.py \
--data_dir './data/processed/full/' \
--train_data train.json \
--dev_data dev.json \
--dev_full dev.full \
--test_data test.json \
--test_full test.full \
--covec_path 'data/MT-LSTM.pt' \
--meta './data/reddit_meta.pick' \
--eval_step 5000  \
--optimizer adam  \
--learning_rate 0.0005 \
--temperature 0.8 \
--if_train 1 \
--contextual_cell_type gru \
--contextual_num_layers 1 \
--msum_cell_type gru \
--msum_num_layers 1 \
--deep_att_lexicon_input_on \
--no_pos --no_ner --no_feat \
--pwnn_on \
--no_lr_scheduler  \
--scheduler_type rop \
--lr_gamma 0.5 \
--log_per_updates 50 \
--decoding_bleu_normalize \
--decoding weight \
--decoding_topk 20 \
--weight_type 'nist' \
--decoding_bleu_lambda 0 \
--test_output 'submission' \
--grad_clipping 0.5 \
--model_type 'BERT' \
--epoches 3  \
--batch_size 12  \
--max_doc 230  \
--max_query 40



##!/bin/bash
#batch_size=$1
#max_doc=$2
#max_query=$3
#epoches=$4
#
#if [ ! -d "log" ]; then
#  mkdir log
#fi
#CUDA_VISIBLE_DEVICES=1 \
#python3  train_stat.py \
#--data_dir 'data/processed/toy/' \
#--train_data train.json \
#--dev_data dev.json \
#--dev_full dev.full \
#--test_data test.json \
#--test_full test.full \
#--covec_path 'data/MT-LSTM.pt' \
#--meta './data/reddit_meta.pick' \
#--eval_step 100  \
#--optimizer adam  \
#--learning_rate 0.0005 \
#--temperature 0.8 \
#--if_train 1 \
#--contextual_cell_type gru \
#--contextual_num_layers 1 \
#--msum_cell_type gru \
#--msum_num_layers 1 \
#--deep_att_lexicon_input_on \
#--no_pos --no_ner --no_feat \
#--pwnn_on \
#--no_lr_scheduler  \
#--scheduler_type rop \
#--lr_gamma 0.5 \
#--log_per_updates 50 \
#--decoding_bleu_normalize \
#--decoding weight \
#--decoding_topk 20 \
#--weight_type 'nist' \
#--decoding_bleu_lambda 0 \
#--test_output 'submission' \
#--grad_clipping 0.5 \
#--model_type 'BERT'  \
#--batch_size $batch_size  \
#--max_doc $max_doc \
#--max_query $max_query  \
#--epoches $epoches \
#
