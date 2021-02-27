#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 1/15/21
#

import argparse
import os
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from cmr_Dataset import CMRDataset

class CMRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            module_args: argparse.Namespace,
    ):
        super().__init__()
        self.train_df_path = os.path.join(module_args.data_dir, module_args.train_tsv)
        self.val_df_path = os.path.join(module_args.data_dir, module_args.val_tsv)
        self.test_df_path = os.path.join(module_args.data_dir, module_args.test_tsv)
        self.tokenizer = T5Tokenizer.from_pretrained(module_args.tokenizer_name_or_path)
        self.tokenizer.add_tokens(['<p>', '</p>', '<title>', '</title>', '<h>', '</h>', '<NUM>', "<TRNC>", "{", "}"])
        self.args = module_args

    def setup(self):
        self.train_dataset = CMRDataset(
            data_path=self.train_df_path,
            tokenizer=self.tokenizer,
            max_query=self.args.max_query,
            max_doc=self.args.max_doc,
            max_answer=self.args.max_answer,
            source_max_token_len=self.args.source_max_token_len,
            target_max_token_len=self.args.target_max_token_len
        )

        print("self.val_df_path: ", self.val_df_path)
        self.val_dataset = CMRDataset(
            data_path=self.val_df_path,
            tokenizer=self.tokenizer,
            max_query=self.args.max_query,
            max_doc=self.args.max_doc,
            max_answer=self.args.max_answer,
            source_max_token_len=self.args.source_max_token_len,
            target_max_token_len=self.args.target_max_token_len
        )

        print("self.test_df_path: ", self.test_df_path)
        self.test_dataset = CMRDataset(
            data_path=self.test_df_path,
            tokenizer=self.tokenizer,
            max_query=self.args.max_query,
            max_doc=self.args.max_doc,
            max_answer=self.args.max_answer,
            source_max_token_len=self.args.source_max_token_len,
            target_max_token_len=self.args.target_max_token_len
        )

        self.steps_per_epoch = len(self.train_dataset) // (self.args.train_batch_size * max(1, len(self.args.gpus)))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True # training on 1 GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=2,
            pin_memory=True  # training on 1 GPU
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=2,
            pin_memory=True  # training on 1 GPU
        )