#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 1/26/21
#

import pandas as pd

def main():
    train2_cleaned_tsv = "data/processed/full/train2_cleaned.tsv"
    dev2_cleaned_tsv = "data/processed/full/dev2_cleaned.tsv"
    test2_tsv = "data/processed/full/test2.tsv"

    train_cleaned_full = "data/processed/full/train_cleaned.full"
    dev_cleaned_full = "data/processed/full/dev_cleaned.full"
    test_full = "data/processed/full/test.full"


    train2_cleaned_tsv = pd.read_csv(train2_cleaned_tsv, sep='\t')
    dev2_cleaned_tsv = pd.read_csv(dev2_cleaned_tsv, sep='\t')
    test2_tsv = pd.read_csv(test2_tsv, sep='\t')


    print("len train2_cleaned_tsv: ", len(train2_cleaned_tsv))
    with open(train_cleaned_full,  'r', encoding='utf8') as train_cleaned_full:
      train_cleaned_full = len(train_cleaned_full.readlines())
      print("train_cleaned_full: ", train_cleaned_full)


    print("len dev2_cleaned_tsv: ", len(dev2_cleaned_tsv))
    with open(dev_cleaned_full,  'r', encoding='utf8') as dev_cleaned_full:
      dev_cleaned_full = len(dev_cleaned_full.readlines())
      print("dev_cleaned_full: ", dev_cleaned_full)


    print("len test2_tsv: ", len(test2_tsv))
    with open(test_full,  'r', encoding='utf8') as test_full:
      test_full = len(test_full.readlines())
      print("test_full: ", test_full)

if __name__ == '__main__':
    main()