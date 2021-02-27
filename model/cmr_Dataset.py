#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 1/15/21
#

import re
import pandas as pd
from torch.utils.data import Dataset

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


class CMRDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer: T5Tokenizer,
            max_query: int,
            max_doc: int,
            max_answer: int,
            source_max_token_len: int = 400,
            target_max_token_len: int = 60,

    ):
        self.data = pd.read_csv(data_path, sep='\t')
        self.tokenizer = tokenizer

        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.max_answer = max_answer
        self.max_query = max_query
        self.max_doc = max_doc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        # Get query, doc, answer tokens
        query_tok = data_row['query_tok']  # text
        doc_tok = data_row['doc_tok']  # text
        answer_tok = data_row['answer_tok']  # text

        # Replace some tokens
        old_list = ["UNK", "<h<NUM>>", "</h<NUM>>"]
        new_list = ["unk", "<h>", "</h>"]
        for old, new in zip(old_list, new_list):
            query_tok = re.sub(old, new, query_tok)
            doc_tok = re.sub(old, new, doc_tok)
            answer_tok = re.sub(old, new, answer_tok)

        # Get tokens with specified length for query, doc, answer
        query_tok = " ".join(
            query_tok.split()[:min(self.max_query, len(query_tok.split()))])  # max query tok length is self.max_query
        doc_tok = " ".join(
            doc_tok.split()[:min(self.max_doc, len(doc_tok.split()))])  # max doc tok length is self.max_doc
        answer_tok = " ".join(answer_tok.split()[:min(self.max_answer, len(
            answer_tok.split()))])  # max answer tok length is self.max_answer

        # print("query_tok: ", query_tok)
        # print("doc_tok: ", doc_tok)
        source_encoding = self.tokenizer(
            query_tok,  # 40
            doc_tok,  # 200
            max_length=self.source_max_token_len,  # 400
            padding='max_length',
            truncation="only_second",  # need to experiement
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            answer_tok,
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            query=query_tok,
            doc=doc_tok,
            answer=answer_tok,
            source_ids=source_encoding["input_ids"].flatten(),
            source_mask=source_encoding['attention_mask'].flatten(),
            target_mask=target_encoding['attention_mask'].flatten(),
            target_ids=labels.flatten()
        )
