import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from densephrases import DensePhrases
from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from simcse import SimCSE
from torch import nn, optim, sum as tsum, reshape
from torch.utils.data import DataLoader, Dataset

from utils_phrasebert import load_model
from datetime import datetime









    all_phrase_embs = []
    if isinstance(model, DensePhrases):
        max_seq_length = model.config.max_position_embeddings
    elif isinstance(model, SimCSE):
        max_seq_length = model.model.config.max_position_embeddings
    else:
        max_seq_length = model.get_max_seq_length()

    for idx, (phrase, sent) in enumerate(zip(list_phrase, sentences)):
        encoded_phrase = model.tokenizer.encode_plus(text=phrase, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
        encoded_phrase = np.array(encoded_phrase["input_ids"])[np.array(encoded_phrase["attention_mask"]) == 1]

        encoded_sent = model.tokenizer.encode_plus(text=sent, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
        encoded_sent = np.array(encoded_sent["input_ids"])[np.array(encoded_sent["attention_mask"]) == 1]

        try:
            start_idx, end_idx = find_sub_list(list(encoded_phrase[1:-1]), list(encoded_sent))
            if end_idx >= max_seq_length:
                print("Context is too long: Idx {} - Phrase: {} - Sentence: {}".format(idx, phrase, sent))
                all_phrase_embs.append(torch.FloatTensor(0))
                continue
        except:
            print("Phrase not found: Idx {} - Phrase: {} - Sentence: {}".format(idx, phrase, sent))
            all_phrase_embs.append(torch.FloatTensor(0))
            continue

        phrase_indices = list(range(start_idx, end_idx + 1, 1))

        if isinstance(model, DensePhrases):
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)
        elif isinstance(model, SimCSE):
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)
        else:
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)

        all_phrase_embs.append(phrase_embs)

    return all_phrase_embs


class ParaphraseDataset(Dataset):




class ProbingModel(LightningModule):



    








