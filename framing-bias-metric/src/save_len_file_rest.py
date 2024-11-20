#!/usr/bin/env python

import fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer
from utils import Seq2SeqDataset, pickle_save



    train_lens = get_lens(train_ds)
    val_ds = Seq2SeqDataset(tok, data_dir, max_source_length, max_target_length, type_path="val", **kwargs)
    val_lens = get_lens(val_ds)
    pickle_save(train_lens, train_ds.len_file)
    pickle_save(val_lens, val_ds.len_file)


if __name__ == "__main__":
    fire.Fire(save_len_file)