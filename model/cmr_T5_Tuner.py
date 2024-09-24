#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 1/15/21
#

import os
import torch
import pytorch_lightning as pl
from my_utils.squad_eval import get_bleu_moses
import os.path
from my_utils.utils import AverageMeter
from train_util import compute_diversity
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)



class T5FineTuner(pl.LightningModule):
    def __init__(self,
                 args,
                 steps_per_epoch=None):

        super(T5FineTuner, self).__init__()
        self.hparams = args
        self.steps_per_epochs = steps_per_epoch
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        # tokenizer
        self.tokenizer.add_tokens(['<p>', '</p>', '<title>', '</title>', '<h>', '</h>', '<NUM>', "<TRNC>", "{", "}"])
        self.source_max_token_len = self.hparams.source_max_token_len
        self.target_max_token_len = self.hparams.target_max_token_len
        self.batch_step = 0
        self.updates = 0
        self.epoch_train_loss = 0
        self.acc_train_step_loss = 0

        '''
        About loss acumulating the loss across all steps
        '''
        self.train_loss = AverageMeter()  # ??
        self.general_dev_loss = AverageMeter()  # ??

        if not os.path.exists(args.output_path):
            print("args.output_path: ", args.output_path)
            os.makedirs(args.output_path)
        self.curve_file = os.path.join(self.hparams.output_path, args.curve_file)
        print("curve_file if train: ", self.curve_file)
        with open(self.curve_file, 'w', encoding="utf8") as fout_dev:
            fout_dev.write('{0},{1},{2},{3},{4},'
                           '{5},{6},{7},{8},{9},{10},{11}\n'.format("epoch",
                                                                    "batch",
                                                                    "global_step",
                                                                    "model.train_loss.avg",
                                                                    "self.epoch_train_loss",
                                                                    "dev_loss",
                                                                    "val_loss",
                                                                    "general_dev_loss",
                                                                    "float(bleu)",
                                                                    "float(bleu_fact)",
                                                                    "float(diver_uni)",
                                                                    "float(diver_bi)")
                                                                    )

    def is_logger(self):
        return self.trainer.global_rank <= 0
    # def is_finished(self):
    #     return self.trainer.global_rank ==


    def forward(
            self,
            input_ids,
            attention_mask=None,
            # decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,

        )

    def _step(self, batch):
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=batch["target_ids"]
        )
        # loss = outputs[0]
        return outputs

    # Training
    def training_step(self, batch, batch_idx):

        loss = self._step(batch)[0]  # torch tensor
        # print("type_ loss: ", type(loss), loss.item(), type(loss.item()))
        self.updates += 1
        self.acc_train_step_loss += loss.item()  #  loss.item()  float
        self.epoch_train_loss = self.acc_train_step_loss / (batch_idx + 1)
        self.batch_step = batch_idx
        self.train_loss.update(loss.item(), len(batch["answer"]))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("self.train_loss.avg", self.train_loss.avg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, " self.train_loss": self.train_loss.avg}  # "log": tensorboard_logs

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_train_loss = avg_train_loss
        self.acc_train_step_loss = 0
        self.log("avg_train_loss", avg_train_loss, prog_bar=True, logger=True)

    # Vaidation
    def validation_step(self, batch, batch_idx):
        val_result = self.generate_step(batch, "val_loss")
        self.general_dev_loss.update(val_result['val_loss'], val_result["batch_length"])
        self.log("self.general_dev_loss", self.general_dev_loss.avg, on_step=True, on_epoch=True, prog_bar=True, logger=True)#, sync_dist=True)

        return val_result

    def validation_epoch_end(self, val_step_outputs):
        # if self.is_logger():
            print("end validation")
            print("self.trainer.global_rank: ", self.trainer.global_rank)
            avg_loss = torch.stack([x["val_loss"] for x in val_step_outputs]).mean()
            total_loss = torch.stack([x["val_loss"] * x["batch_length"] for x in val_step_outputs]).sum()
            data_len = sum([x["batch_length"] for x in val_step_outputs])
            val_loss_acc = total_loss / data_len

            answers = []
            docs = []
            preds = []

            for x in val_step_outputs:
                answers += x['answer']
            for x in val_step_outputs:
                docs += x['doc']
            for x in val_step_outputs:
                preds += x['pred']

            answers_tok_list = [line.strip().split(' ') for line in answers]
            docs_tok_list = [line.strip().split(' ') for line in docs]
            pred_tok_list = [line.strip().split(' ') for line in preds]

            bleu_result = get_bleu_moses(pred_tok_list, answers_tok_list)
            bleu = str(bleu_result).split('=')
            bleu = bleu[1].split(',')[0]

            bleu_fact = get_bleu_moses(docs_tok_list, pred_tok_list)
            bleu_fact = str(bleu_fact).split('=')
            bleu_fact = bleu_fact[1].split(',')[0]

            pred_output_dir = os.path.join(self.hparams.output_path, 'pred/dev/')
            if not os.path.exists(pred_output_dir):
                os.makedirs(pred_output_dir)
            full_pred_output_dir = os.path.join(self.hparams.output_path, 'full_pred/dev/')
            if not os.path.exists(full_pred_output_dir):
                os.makedirs(full_pred_output_dir)

            pred_output_path = os.path.join(pred_output_dir, str(self.trainer.global_step)) + '.txt'
            full_pred_output_path = os.path.join(full_pred_output_dir, str(self.trainer.global_step)) + '_full.txt'

            with open(pred_output_path, 'w') as f:
                for hypothesis in pred_tok_list:
                    f.write(' '.join(hypothesis) + '\n')

            full_file_path = os.path.join(self.hparams.data_dir, self.hparams.dev_full)
            with open(full_file_path, 'r', encoding='utf8') as fr:
                full_lines = fr.readlines()
                print("len(full_lines) :", len(full_lines))
                print("pred_words_list: ", len(pred_tok_list))
                print("full_file_path: ", full_file_path)
                assert (len(full_lines) == len(pred_tok_list))
                with open(full_pred_output_path, 'w', encoding='utf8') as fw:
                    for f, g in zip(full_lines, pred_tok_list):
                        f = f.strip().split('\t')
                        f[-1] = ' '.join(g).strip()
                        f = '\t'.join(f)
                        fw.write(f + '\n')
                        fw.flush()

            diversity = compute_diversity(pred_tok_list, pred_output_path)  # how comes? output already has pred_words_list
            diversity = str(diversity).strip().split()
            diver_uni = diversity[0][2:]
            diver_bi = diversity[1][:-3]

            with open(self.curve_file, 'a+', encoding="utf8") as fout_dev:
                fout_dev.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f},{6:.5f},'
                               '{7:.5f},{8:.5f},{9:.5f},{10:.5f},{11:.5f}\n'.format(self.current_epoch,
                                                                           self.batch_step,
                                                                           self.trainer.global_step,
                                                                           self.train_loss.avg,
                                                                           self.epoch_train_loss,
                                                                           avg_loss,
                                                                           val_loss_acc,
                                                                           self.general_dev_loss.avg ,
                                                                           float(bleu),
                                                                           float(bleu_fact),
                                                                           float(diver_uni),
                                                                           float(diver_bi)
                                                                           )
                               )

            curve_df = pd.read_csv(self.curve_file, encoding="utf-8")
            curve_df.plot(x="epoch", y=["model.train_loss.avg", "self.epoch_train_loss", "dev_loss", "val_loss","general_dev_loss"],
                          title="T5 loss")
            fig_path = self.hparams.output_path + '/epoch_T5_loss.png'
            plt.savefig(fig_path)

            curve_df.plot(x="batch",
                          y=["model.train_loss.avg", "self.epoch_train_loss", "dev_loss", "val_loss", "general_dev_loss"],
                          title="T5 loss")
            fig_path = self.hparams.output_path + '/batch_T5_loss.png'
            plt.savefig(fig_path)
            self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)#, sync_dist=True)
            self.log("val_loss_acc", val_loss_acc, prog_bar=True, logger=True)#, sync_dist=True)

            answers.clear()
            docs.clear()
            preds.clear()
            answers_tok_list.clear()
            docs_tok_list.clear()
            pred_tok_list.clear()



    # Testing
    def test_step(self, batch, batch_idx):
        return self.generate_step(batch, "test_loss")

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x["test_loss"] for x in test_step_outputs]).mean()
        answers = []
        docs = []
        preds = []

        for x in test_step_outputs:
            answers += x['answer']
        for x in test_step_outputs:
            docs += x['doc']
        for x in test_step_outputs:
            preds += x['pred']

        answers_tok_list = [line.strip().split(' ') for line in answers]
        docs_tok_list = [line.strip().split(' ') for line in docs]
        pred_tok_list = [line.strip().split(' ') for line in preds]

        dstc = os.path.join(self.hparams.output_path, 'dstc')
        if not os.path.exists(dstc):
            os.makedirs(dstc)

        pred_output_dir = os.path.join(self.hparams.output_path, 'pred/test/')
        if not os.path.exists(pred_output_dir):
            os.makedirs(pred_output_dir)
        full_pred_output_dir = os.path.join(self.hparams.output_path, 'full_pred/test/')
        if not os.path.exists(full_pred_output_dir):
            os.makedirs(full_pred_output_dir)

        pred_output_path = os.path.join(pred_output_dir, str(self.trainer.global_step)) + '.txt'
        full_pred_output_path = os.path.join(full_pred_output_dir, str(self.trainer.global_step)) + '_full.txt'

        with open(pred_output_path, 'w') as f:
            for hypothesis in pred_tok_list:
                f.write(' '.join(hypothesis) + '\n')

        full_file_path = os.path.join(self.hparams.data_dir, self.hparams.test_full)
        with open(full_file_path, 'r', encoding='utf8') as fr:
            full_lines = fr.readlines()
            print("len(full_lines) :", len(full_lines))
            print("pred_words_list: ", len(pred_tok_list))
            print("full_file_path: ", full_file_path)
            assert (len(full_lines) == len(pred_tok_list))
            with open(full_pred_output_path, 'w', encoding='utf8') as fw:
                for f, g in zip(full_lines, pred_tok_list):
                    f = f.strip().split('\t')
                    f[-1] = ' '.join(g).strip()
                    f = '\t'.join(f)
                    fw.write(f + '\n')
                    fw.flush()

        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True) #, sync_dist=True)
        answers.clear()
        docs.clear()
        preds.clear()
        answers_tok_list.clear()
        docs_tok_list.clear()
        pred_tok_list.clear()

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        optimizer = AdamW(self.parameters())

        return [optimizer]  # , [scheduler]

    def generate_step(self, batch, loss_name):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        pred = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ] # text
        loss = self._step(batch).loss
        # print("log val loss:   ", loss_name, loss)
        self.log(loss_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #, sync_dist=True)
        return {loss_name: loss, "batch_length": len(batch['answer']), "pred": pred, "answer": batch['answer'],
                "doc": batch['doc']}

    def collect_results_gpu(self, result_part, size):
        rank, world_size = get_dist_info()
        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            return ordered_results

        # return preds

    # def compute_diversity(self, hypotheses, output_path):  # did not understand!!!!
    #     hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    #     pipe = subprocess.Popen(
    #         ["perl", './bleu_eval/diversity.pl.remove_extension', output_path],
    #         stdin=subprocess.PIPE,
    #         stdout=subprocess.PIPE
    #     )
    #     pipe.stdin.write(hypothesis_pipe.encode())
    #     pipe.stdin.close()
    #     return pipe.stdout.read()
