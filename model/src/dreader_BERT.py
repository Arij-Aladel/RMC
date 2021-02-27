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
        #TODO: Load BERT tokenizer and 
        # Load the tokenizer for our model.
        #TODO: CHANGED TO DistilBERT #"bert-base-uncased"
        print("\nLoading BERT tokenizer...")
        self.hidden_size = 768

        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', #'distilbert-base-uncased',
            do_lower_case=True
        )

        #TODO: CHANGED TO DistilBERT #"bert-base-uncased"
        print("\nLoading pretrained BertForQuestionAnswering")
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            'bert-base-uncased', # 'distilbert-base-uncased',
            output_attentions = False, # Return attentions weights or not
            output_hidden_states = True, # Returns all hidden-states or not
        )

        
    def forward(self, batch, vocab):
        max_len = 512     # max_len of BERT input
        fix_doc_len = self.opt['max_doc']  # fix lenght of document 
        fix_q_len = max_len - fix_doc_len - 3#self.opt['max_len']   # fix length of conversation history (query)
        batch_size = len(batch['query_tok'])

        all_input_ids = []
        attention_masks = []
        segment_ids = []

        doc_mask = torch.LongTensor(batch_size, fix_doc_len).fill_(1)  # for SAN decoder
        for i in range(batch_size):

            #print("[int(j.item()) for j in batch['doc_tok'][i]]: ",[int(j.item()) for j in batch['doc_tok'][i]])
            doc_text = " ".join([vocab[int(j.item())] if int(j.item()) else '' for j in batch['doc_tok'][i]])
            doc_tokens = self.tokenizer.tokenize(doc_text)
            doc_tokens = doc_tokens[:fix_doc_len] # truncating
            # while len(doc_tokens)<fix_doc_len: #TODO: ADDEDBY Hamza  # padding but I delete it since bert will pad at the end
            #   doc_tokens.append('[PAD]')
            # print("\n Original doc tokens length: ", len([j.item() for j in batch['doc_tok'][i] if j.item()>0 ]))
            # print("BERT doc tokens length: ", len(doc_tokens))  
            doc_text = " ".join(doc_tokens)
            # build doc mask
            doc_mask[i, :len(doc_tokens)] = torch.LongTensor([0]*len(doc_tokens))


            q_text = " ".join([vocab[int(j.item())] if int(j.item()) else '' for j in batch['query_tok'][i]])
            q_tokens = self.tokenizer.tokenize(q_text)
            q_tokens = q_tokens[:fix_q_len] # truncating
            while len(q_tokens)<fix_q_len: # padding
              q_tokens.append('[PAD]')
            # print("Original query tokens length: ", len([j.item() for j in batch['query_tok'][i]   if j.item()>0  ]))  
            # print("BERT query tokens length: ", len(q_tokens))  
            q_text = " ".join(q_tokens)

            # # Apply the tokenizer to the input text, treating them as a text-pair.
            # input_ids = tokenizer.encode(q_text, doc_text)
            # tokens = tokenizer.convert_ids_to_tokens(input_ids)
            encoded_dict = self.tokenizer.encode_plus(
            q_text, 
            doc_text,
            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
            max_length = max_len,       # Pad & truncate all sentences.
            pad_to_max_length = True,
            truncation = True,
            return_attention_mask = True, 
            return_tensors = 'pt',        
            )

            all_input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])    
            segment_ids.append(encoded_dict['token_type_ids'])


            # # For each token and its id...
            # for tokens, id in zip(tokens, input_ids):
                
            #     # If this is the [SEP] token, add some space around it to make it stand out.
            #     if id == tokenizer.sep_token_id:
            #         print('')
                
            #     # Print the token string and its ID in two columns.
            #     print('{:<12} {:>6,}'.format(token, id))

            #     if id == tokenizer.sep_token_id:
            #         print('')
    

            # # Search the input_ids for the first instance of the `[SEP]` token.
            # sep_index = input_ids.index(tokenizer.sep_token_id)

            # # The number of segment A tokens includes the [SEP] token istelf.
            # num_seg_a = sep_index + 1

            # # The remainder are segment B.
            # num_seg_b = len(input_ids) - num_seg_a

            # # Construct the list of 0s and 1s.
            # seg_ids = [0]*num_seg_a + [1]*num_seg_b

            # # There should be a segment_id for every input token.
            # # assert len(seg_ids) == len(input_ids)    

            # all_input_ids.append(input_ids)
            # segment_ids.append(seg_ids)

        #attention_masks = [[float(i>0) for i in seq] for seq in all_input_ids]
        all_input_ids = torch.cat(all_input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)


        if self.opt['cuda']: 
            all_input_ids = self.patch(all_input_ids) #all_input_ids.cuda()
            attention_masks = self.patch(attention_masks) #attention_masks.cuda()
            segment_ids = self.patch(segment_ids) #segment_ids.cuda()

        #self.bert_model.zero_grad()    #???????????    

        print("all_input_ids: ", all_input_ids.shape)
        print("attention_masks: ", attention_masks.shape)
        print("segment_ids: ", segment_ids.shape)
        outputs = self.bert_model(all_input_ids, 
                             attention_mask=attention_masks, 
                             token_type_ids = segment_ids,
                             output_hidden_states=True)
        return
        hidden_states = outputs[2][0]
        if self.opt['cuda']: 
            hidden_states = self.patch(hidden_states) #hidden_states.cuda() # shape (8, 512, 768) -> (batch, input_size, embedding_size)

        # Extract query, document hidden states and oc_mask
        # query
        query_mem = []
        for i in range(batch_size):
            h_state = hidden_states[i]
            q_mem = h_state[1:fix_q_len+1] # (fix_q_len, 768)
            q_mem = torch.mean(q_mem, 0).tolist() # (1, 768)
            query_mem.append(q_mem)
        query_mem = torch.Tensor(query_mem)
        if self.opt['cuda']: 
            query_mem = self.patch(query_mem)
        
        # document
        doc_mem = []
        for i in range(batch_size):
            h_state = hidden_states[i]
            #d_mem = h_state[fix_q_len+2:fix_q_len+fix_doc_len+2]
            d_mem = h_state[fix_q_len+2:-1].tolist() # (d_len, 768) # TODO: MODIFIED TO NEEW SIZE
            doc_mem.append(d_mem)
        doc_mem = torch.Tensor(doc_mem)
        if self.opt['cuda']:
            doc_mem = self.patch(doc_mem) # (batch_size, d_len, 768))
        # doc_mask
        doc_mask = torch.tensor(doc_mask, dtype=torch.bool)  # (batch_size, d_len)
        if self.opt['cuda']:
            doc_mask = self.patch(doc_mask)

        # print("doc_mask from BERT: ", doc_mask.shape)
        # print("query_mem from BERT encoder: ", query_mem.shape)
        # print("doc_mem from BERT encoder: ", doc_mem.shape)

        return doc_mem, query_mem, doc_mask

    def patch(self, v):
        if self.opt['cuda']:
            v = Variable(v.cuda(non_blocking=True))
        else:
            v = Variable(v)
        return v
