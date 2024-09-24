#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/9/20
#


import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .recurrent import OneLayerBRNN, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .encoder import LexiconEncoder
from .similarity import DeepAttentionWrapper, FlatSimilarityWrapper, SelfAttnWrapper
from .similarity import AttentionWrapper
from .san_decoder import SANDecoder
from transformers import BertTokenizer, BertForQuestionAnswering


class DNetwork_BERT(nn.Module):
    """enoder using BERT"""

    def __init__(self, opt, embedding=None, padding_idx=0):
        super(DNetwork_BERT, self).__init__()
        # TODO: Load BERT tokenizer and
        # Load the tokenizer for our model.
        # TODO: CHANGED TO DistilBERT #"bert-base-uncased"
        print("\nLoading BERT tokenizer...")
        self.hidden_size = 768

        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',  # 'distilbert-base-uncased',
            do_lower_case=True
        )

        # TODO: CHANGED TO DistilBERT #"bert-base-uncased"
        print("\nLoading pretrained BertForQuestionAnswering")
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            'bert-base-uncased',  # 'distilbert-base-uncased',
            output_attentions=False,  # Return attentions weights or not
            output_hidden_states=True,  # Returns all hidden-states or not
        )

    def forward(self, batch, vocab):

        trunc_query = []
        trunc_doc = []

        max_len = 512  # max_len of BERT input
        fix_d_len = self.opt['max_doc'] + 170  # fix lenght of document
        if (fix_d_len>=max_len-3):
          print("Max length of document should not exceed max input of BERT: max_doc = {}, BERT max input ={}".format(self.opt['max_doc'], max_len))
          exit()
        fix_q_len = max_len - fix_d_len - 3  # self.opt['max_len']   # fix length of conversation history (query)
        batch_size = len(batch['query_tok'])

        all_input_ids = []
        attention_masks = []
        segment_ids = []
        ######TODO: ADDED
        loss_query = 0  # num of truncated queries
        loss_doc = 0  # number of truncated docs
        ############

        doc_mask = torch.LongTensor(batch_size, fix_d_len).fill_(1)  # for SAN decoder
        for i in range(batch_size):

            doc_text = [vocab[int(j.item())] for j in batch['doc_tok'][i] if int(j.item())]
            doc_text = " ".join(doc_text)

            q_text = [vocab[int(j.item())] for j in batch['query_tok'][i] if int(j.item())]
            q_text = " ".join(q_text)

            input_ids = self.tokenizer.encode(q_text, doc_text)

            # Search the input_ids for the first instance of the `[SEP]` token.
            sep_index = input_ids.index(self.tokenizer.sep_token_id)

            mask_q = [1] * (fix_q_len + 2)  # attention mask for query
            num_seg_q = sep_index + 1
            num_seg_d = len(input_ids) - num_seg_q

            input_ids_q = input_ids[:num_seg_q]  # [CLS]query[SEP]
            input_ids_d = input_ids[num_seg_q:]

            # Query: build attention_mask, document_ids and segments_ids
            if len(input_ids_q) > fix_q_len + 2:
                # trunc_query.extend(input_ids_q[fix_q_len+1:])
                loss_query += 1
                input_ids_q = [input_ids_q[0]] + input_ids_q[1:fix_q_len + 1] + [input_ids_q[-1]]
                # mask_q = mask_q[:fix_q_len + 2]
            elif len(input_ids_q) < fix_q_len + 2:
                mask_q = mask_q[:len(input_ids_q)] + [0] * (fix_q_len - num_seg_q + 2)
                input_ids_q = input_ids_q + [0] * (fix_q_len - num_seg_q + 2)
            num_seg_q = fix_q_len + 2
            segments_ids_q = [0] * num_seg_q  # Query Segment_ids

            # Document: build attention_mask, qocument_ids and segments_ids
            mask_d = [1] * (fix_d_len + 1)
            if len(input_ids_d) > fix_d_len + 1:
                # trunc_doc.extend(input_ids_d[fix_d_len:])
                loss_doc += 1
                input_ids_d = input_ids_d[:fix_d_len] + [input_ids_d[-1]]  # input ids for document

            elif len(input_ids_d) < fix_d_len + 1:
                mask_d = mask_d[:len(input_ids_d)] + [0] * (fix_d_len - len(input_ids_d) + 1)
                input_ids_d = input_ids_d + [0] * (fix_d_len - num_seg_d + 1)  # input ids for document

            num_seg_d = fix_q_len + fix_d_len + 2
            segments_ids_d = [1] * (max_len - num_seg_q)  # Document segment_ids

            # build doc mask
            doc_mask[i] = torch.LongTensor(mask_d[:-1])

            input_ids = input_ids_q + input_ids_d
            seg_ids = segments_ids_q + segments_ids_d
            attention_mask = mask_q + mask_d

            all_input_ids.append(torch.tensor(input_ids).unsqueeze(0))
            attention_masks.append(torch.tensor(attention_mask).unsqueeze(0))
            segment_ids.append(torch.tensor(seg_ids).unsqueeze(0))

        all_input_ids = torch.cat(all_input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)

        if self.opt['cuda']:
            all_input_ids = self.patch(all_input_ids)  # all_input_ids.cuda()
            attention_masks = self.patch(attention_masks)  # attention_masks.cuda()
            segment_ids = self.patch(segment_ids)  # segment_ids.cuda()

        # doc_mask
        doc_mask = torch.eq(doc_mask, 0)
        # doc_mask = torch.tensor(doc_mask, dtype=torch.bool)  # (batch_size, d_len)
        if self.opt['cuda']:
            doc_mask = self.patch(doc_mask)

        outputs = self.bert_model(all_input_ids,
                                  attention_mask=attention_masks,
                                  token_type_ids=segment_ids,
                                  output_hidden_states=True)

        hidden_states = outputs[2][0]
        if self.opt['cuda']:
            hidden_states = self.patch(
                hidden_states)  # hidden_states.cuda() # shape (8, 512, 768) -> (batch, input_size, embedding_size)

        # Extract query, document hidden states and oc_mask
        # query
        query_mem = []
        for i in range(batch_size):
            h_state = hidden_states[i]
            q_mem = h_state[1:fix_q_len + 1]  # (fix_q_len, 768)
            q_mem = torch.mean(q_mem, 0).tolist()  # (1, 768)
            query_mem.append(q_mem)
        query_mem = torch.Tensor(query_mem)
        if self.opt['cuda']:
            query_mem = self.patch(query_mem)

        # document
        doc_mem = []
        for i in range(batch_size):
            h_state = hidden_states[i]
            # d_mem = h_state[fix_q_len+2:fix_q_len+fix_doc_len+2]
            d_mem = h_state[fix_q_len + 2:-1].tolist()  # (d_len, 768) # TODO: MODIFIED TO NEEW SIZE
            doc_mem.append(d_mem)
        doc_mem = torch.Tensor(doc_mem)
        if self.opt['cuda']:
            doc_mem = self.patch(doc_mem)  # (batch_size, d_len, 768))


        return doc_mem, query_mem, doc_mask, loss_doc, loss_query

    def patch(self, v):
        if self.opt['cuda']:
            v = Variable(v.cuda(non_blocking=True))
        else:
            v = Variable(v)
        return v


