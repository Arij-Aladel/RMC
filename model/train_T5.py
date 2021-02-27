#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 1/24/21
#

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from cmr_T5_Tuner import T5FineTuner
from cmr_DataModule import CMRDataModule

def main():
    # on_step=True, on_epoch=True
    # model arguments
    check_dir = "checkpoints_tuner_with_generator_toy_1"

    # call back
    checkpoint_callback = ModelCheckpoint(
        dirpath=check_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='tuner_with_generator_log_toy_1'
    )

    module_dict = dict(
        # data_dir="./data/processed/toy/",  # path for data files
        # train_tsv='train.tsv',
        # val_tsv='dev.tsv',
        # test_tsv='test.tsv',
        # dev_full='dev.full',
        # test_full='test.full',
        # train_batch_size=8,
        # val_batch_size=8,
        # test_batch_size=8,
        data_dir = "./data/processed/toy/", # path for data files
        train_tsv = 'train.tsv',   # 'train2_cleaned.tsv',
        val_tsv = 'dev.tsv', #'dev2_cleaned.tsv',
        test_tsv = 'test.tsv', #'test2.tsv',
        dev_full = 'dev.full', #'dev_cleaned.full',
        test_full = 'test.full',
        train_batch_size=42,
        val_batch_size=42,
        test_batch_size=16,
        curve_file="curve_file.csv",
        output_dir = "test_output", # path to save the checkpoints
        output_path = check_dir,
        model_name_or_path ='t5-small',
        tokenizer_name_or_path = 't5-small',
        max_doc = 200,
        max_query = 40,
        max_answer = 40,
        target_max_token_len = 60,
        source_max_token_len = 400,
        learning_rate = 1e-4,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gpus = [3],
        num_train_epochs = 10,
        # early_stop_callback=False,
        fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
        # opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        seed=42,
        enable_pl_optimizer=True,
        max_grad_norm=0.2, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default

    )



    # Trainer args
    trainer_dict = dict(
        gpus = module_dict['gpus'],
        max_epochs = module_dict['num_train_epochs'],
        amp_level = 'O1',
        # amp_backend='apex' ,
        # accelerator='ddp',
        val_check_interval = 5000,
        progress_bar_refresh_rate = 100 ,
        checkpoint_callback= checkpoint_callback , #checkpoint_callback,
        precision= 16 if module_dict['fp_16'] else 32,
        logger = logger,
        num_sanity_val_steps=0,
        gradient_clip_val=module_dict['max_grad_norm'],
        #resume_from_checkpoint='checkpoints_tuner_with_generator_full/best-checkpoint.ckpt' ,
    )

    module_args = argparse.Namespace(**module_dict)
    trainer_args = argparse.Namespace(**trainer_dict)
    type(trainer_args)


    pl.seed_everything(module_args.seed)
    # tokenizer = T5Tokenizer.from_pretrained(module_args.tokenizer_name_or_path)
    trainer = pl.Trainer(**trainer_dict)
    #num_added_toks

    data_module = CMRDataModule(module_args)
    data_module.setup()
    model = T5FineTuner(args=module_args,
                        steps_per_epoch=data_module.steps_per_epoch)
    trainer.fit(model, data_module)

    trainer.test()

if __name__ == '__main__':
    main()