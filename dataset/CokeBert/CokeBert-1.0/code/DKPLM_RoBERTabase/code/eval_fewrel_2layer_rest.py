# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import simplejson as json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

#from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.tokenization_roberta import RobertaTokenizer
#from knowledge_bert.modeling_eval import BertForSequenceClassification
from knowledge_bert.modeling_new_n_CLS_comb200_rob import BertForSequenceClassification
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from collections import defaultdict
import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)











    #k = (input_ent_outORin_emb*input_ent_r_emb).cuda()
    #v = input_ent_neighbor_emb.cuda()+k
    #return k,v












print("Load Emb ...")
embed_ent, embed_r = load_knowledge()
ent_neighbor, ent_r, ent_outORin = load_ent_emb_static()
#ent_neighbor, ent_r, ent_outORin = load_ent_emb_dynamic()
print("Finsh loading Emb")



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_ent = input_ent
        self.ent_mask = ent_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())

class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "dev")


    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
                    #print(line['text'][x[1]:x[2]].encode("utf-8"))
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples













class InputFeatures(object):
    """A single set of features of data."""



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""




    @classmethod

class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""







def check_pre(a, b):
    if len(a) < len(b):
        return False
    else:
        #a = [x for x in a if x != 'ĠUCHIJ' and x != 'ĠTG' and x != 'ĠUKIP' and x != 'ĠCLSID']
        a = [x for x in a if x != 'ĠUCHIJ' and  x != 'ĠCLSID']
        for i in range(len(b)):
            if a[i] != b[i]:
                return False
        return True

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""

    label_list = sorted(label_list)
    label_map = {label : i for i, label in enumerate(label_list)}

    entity2id = {}
    with open("../../data/kg_embed/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        t_name = ex_text_a[t[1]:t[2]]
        ###
        #assert 'TG' not in ex_text_a
        #assert 'UCHIJ' not in ex_text_a
        #assert 'UKIP' not in ex_text_a
        if 'UKIP' in ex_text_a:
            print(ex_text_a)
            print("=======")
        assert 'CLSID' not in ex_text_a
        ###
        if h[1] < t[1]:
            #ex_text_a = ex_text_a[:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:]
            #ex_text_a = ex_text_a[:h[1]] + "TG "+h_name+" UKIP" + ex_text_a[h[2]:t[1]] + "UCHIJ "+t_name+" CLSID" + ex_text_a[t[2]:]
            ex_text_a = ex_text_a[:h[1]] + "# "+h_name+" UKIP" + ex_text_a[h[2]:t[1]] + "$ "+t_name+" CLSID" + ex_text_a[t[2]:]
        else:
            #ex_text_a = ex_text_a[:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:]
            #ex_text_a = ex_text_a[:t[1]] + "UCHIJ "+t_name+" CLSID" + ex_text_a[t[2]:h[1]] + "TG "+h_name+" UKIP" + ex_text_a[h[2]:]
            ex_text_a = ex_text_a[:t[1]] + "$ "+t_name+" CLSID" + ex_text_a[t[2]:h[1]] + "# "+h_name+" UKIP" + ex_text_a[h[2]:]

        #print(ex_text_a)
        #exit()
        if h[1] < t[1]:
            h[1] += 2
            h[2] += 2
            t[1] += 6
            t[2] += 6
        else:
            h[1] += 6
            h[2] += 6
            t[1] += 2
            t[2] += 2

        #tokens_a, entities_a = tokenizer.tokenize(ex_text_a, [h, t])
        ######

        ###
        #tokens_a = tokenizer.tokenize(ex_text_a)
        first_token = ex_text_a.split(" ")[0]
        if first_token == "#":
            tokens_a = tokenizer.tokenize(" "+ex_text_a)
        else:
            tokens_a = tokenizer.tokenize(ex_text_a)
        ###

        entities_a = ["UNK"] * len(tokens_a)
        for i in range(len(tokens_a)):
            #if tokens_a[i] == 'Ġ#' or tokens_a[i] == '#':
            #if tokens_a[i] == 'ĠTG' or tokens_a[i] == 'TG':
            if tokens_a[i] == 'Ġ#' or tokens_a[i] == '#':
                entities_a[i+1] = h[0]
                break
        for i in range(len(tokens_a)):
            if tokens_a[i] == 'Ġ$' or tokens_a[i] == '$':
                entities_a[i+1] = t[0]
                break
        ######
        #assert len([x for x in entities_a if x!="UNK"]) == 2

        if len([x for x in entities_a if x!="UNK"]) != 2:
            print(ex_text_a)
            print("--")
            print(tokens_a)
            print("--")
            print(entities_a)
            print("--")
            print(len([x for x in entities_a if x[0]!="UNK"]))
            exit(1)


        tokens_b = None
        if example.text_b:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for x in example.text_b[1] if x[-1]>threshold])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                entities_a = entities_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens = ["<s>"] + tokens_a + ["</s>"]
        ents = ["UNK"] + entities_a + ["UNK"]
        segment_ids = [0] * len(tokens)

        '''
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            ents += entities_b + ["UNK"]
            segment_ids += [1] * (len(tokens_b) + 1)
        '''

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ent = []
        ent_mask = []
        for ent in ents:
            if ent != "UNK" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ent += padding_
        ent_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ent) == max_seq_length
        assert len(ent_mask) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("ents: %s" % " ".join(
                    [str(x) for x in ents]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        ###

        #input_ids_tensor = torch.Tensor(input_ids)
        # '#':10431, 'Ġ#':849, 'Ġ$':68, '$':1629, 'ĠCLSID':50001, 'ĠUKIP':35829
        #if 849 not in input_ids or 68 not in input_ids:
            #print(tokens)
            #print(input_ids)
            #print("====")
        for i,id in enumerate(input_ids):
            if 10431 == id: # 'G#' + 'ĠUKIP'
                #ids_id = input_ids.index(10431) # #=>G#
                input_ids[i] = 849
            if 1629 == id: # 'G$' + 'ĠCLSID'
                #ids_id = input_ids.index(1629) # $=>G$
                input_ids[i] = 68
        if 849 not in input_ids or 68 not in input_ids:
            print(input_ids)
            print("Still no 68 or 849")
            exit()
            continue

        input_ids_tensor = torch.LongTensor(input_ids)
        if len(input_ids_tensor[input_ids_tensor == 849])>1 or len(input_ids_tensor[input_ids_tensor== 68]) >1:
            print("More than one 68 or 849")
            print("===")
            print(ex_text_a)
            print("--")
            print(tokens)
            print("--")
            print(input_ids)
            print("---")
            #exit()

            # 'G#'+'GUKIP'
            if len(input_ids_tensor[input_ids_tensor == 849]) >1 :
                id_G = input_ids.index(849)
                '''
                for j in list(range(id_G+1,len(input_ids))):
                    if input_ids[j] == 849:
                        input_ids[j] = 0
                '''
                j = id_G
                locate = list()
                while j < len(input_ids):
                    if input_ids[j] == 35829:
                        #set all 'G#' to 0 but the last one
                        locate = locate[-1]
                        z=0
                        while z < len(input_ids):
                            if input_ids[z] == 849:
                                input_ids[z]=0
                            z+=1
                        input_ids[locate] = 849
                        break
                    else:
                        if input_ids[j] == 849:
                            locate.append(j)
                    j+=1

            # 'G$'+'GCLSID'
            if len(input_ids_tensor[input_ids_tensor == 68]) >1 :
                id_G = input_ids.index(68)
                '''
                for j in list(range(id_G+1,len(input_ids))):
                    if input_ids[j] == 68:
                        input_ids[j] = 0
                '''
                j = id_G
                locate = list()
                locate.append(j)
                while j < len(input_ids):
                    if input_ids[j] == 50001:
                        #set all 'G#' to 0 but the last one
                        locate = locate[-1]
                        z=0
                        while z < len(input_ids):
                            if input_ids[z] == 68:
                                input_ids[z] = 0
                            z+=1
                        input_ids[locate] = 68
                        break
                    else:
                        if input_ids[j] == 68:
                            locate.append(j)
                    j+=1
            print(input_ids)
            print("===")
            print("---")
            #continue
        ###

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_ent=input_ent,
                              ent_mask=ent_mask,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0





def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--threshold', type=float, default=.3)
    ##########ADD##
    parser.add_argument("--K_V_dim",
                        type=int,
                        default=100,
                        help="Key and Value dim == KG representation dim")

    parser.add_argument("--Q_dim",
                        type=int,
                        default=768,
                        help="Query dim == Bert six output layer representation dim")
    parser.add_argument('--graphsage',
                        default=False,
                        action='store_true',
                        help="Whether to use Attention GraphSage instead of GAT")
    parser.add_argument('--self_att',
                        default=True,
                        action='store_true',
                        help="Whether to use GAT")
    parser.add_argument('--data_token',
                        type=str,
                        default='None',
                        help="Using token ids")
    ###############

    args = parser.parse_args()

    processors = FewrelProcessor

    num_labels_task = 80

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    processor = processors()
    num_labels = num_labels_task
    label_list = None

    #tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    tokenizer = RobertaTokenizer.from_pretrained(args.ernie_model)

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    '''
    vecs = []
    vecs.append([0]*100)
    with open("kg_embed/entity2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    #embed = torch.nn.Embedding(5041175, 100)

    logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs
    '''

    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if "pytorch_model.bin_" in x]

    ###
    #filenames = [x for x in filenames if x in ["pytorch_model.bin_1750", "pytorch_model.bin_2000", "pytorch_model.bin_2250", "pytorch_model.bin_2500", "pytorch_model.bin_2750", "pytorch_model.bin_3000", "pytorch_model.bin_3250", "pytorch_model.bin_3500", "pytorch_model.bin_3750", "pytorch_model.bin_4000", "pytorch_model.bin_4250", "pytorch_model.bin_4500", "pytorch_model.bin_4750", "pytorch_model.bin_5000"] ]

    filenames = [x for x in filenames if x in ["pytorch_model.bin_1750", "pytorch_model.bin_2000", "pytorch_model.bin_2250", "pytorch_model.bin_2500", "pytorch_model.bin_2750", "pytorch_model.bin_3000", "pytorch_model.bin_3250", "pytorch_model.bin_3500", "pytorch_model.bin_3750", "pytorch_model.bin_4000"] ]
    ###

    file_mark = []
    for x in filenames:
        file_mark.append([x, True])
        file_mark.append([x, False])

    eval_examples = processor.get_dev_examples(args.data_dir)
    dev = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.threshold)
    eval_examples = processor.get_test_examples(args.data_dir)
    test = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.threshold)

    for x, mark in file_mark:
        print(x, mark)
        #if mark == True: ###erine
        #    continue ###erine
        #else: ###erine
        #    print("Test") ###erine
        #exit()
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        #model, _ = BertForSequenceClassification.from_pretrained(args.ernie_model, state_dict=model_state_dict, num_labels=len(label_list))
        model, _ = BertForSequenceClassification.from_pretrained(args.ernie_model, state_dict=model_state_dict, num_labels=len(label_list), args=args)

        #model.to(device)
        #print(device)

        if args.fp16: #
            model.half() #
        model.to(device)

        #print(model)
        #print(list(model.named_parameters()))
        #print("==")
        #print(list(model.bert.word_graph_attention.K_V_linear.weight))
        #exit()
        #for i in model.parameters():
        #    print(i)
        #exit()
        #for name, param in model.named_parameters():
        #    print(name,param.requires_grad)


        if mark:
            eval_features = dev
        else:
            eval_features = test
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # zeros = [0 for _ in range(args.max_seq_length)]
        # zeros_ent = [0 for _ in range(100)]
        # zeros_ent = [zeros_ent for _ in range(args.max_seq_length)]
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        #all_text = torch.tensor([f.text for f in eval_features], dtype=torch.long)
        all_ent = torch.tensor([f.input_ent for f in eval_features], dtype=torch.long)
        all_ent_masks = torch.tensor([f.ent_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ent, all_ent_masks, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if mark:
            output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(x.split("_")[-1]))
            output_file_pred = os.path.join(args.output_dir, "eval_pred_{}.txt".format(x.split("_")[-1]))
            output_file_glod = os.path.join(args.output_dir, "eval_gold_{}.txt".format(x.split("_")[-1]))
        else:
            output_eval_file = os.path.join(args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))
            output_file_pred = os.path.join(args.output_dir, "test_pred_{}.txt".format(x.split("_")[-1]))
            output_file_glod = os.path.join(args.output_dir, "test_gold_{}.txt".format(x.split("_")[-1]))

        fpred = open(output_file_pred, "w")
        fgold = open(output_file_glod, "w")

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        #embed_ent, embed_r, e1_e2_r_outORin = load_knowledge()

        #logits_erine = torch.load('logits.pt') ###erine
        #i_index = 0 ###erine
        for input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids in eval_dataloader:
            #input_ent = embed(input_ent+1) # -1 -> 0
            ###
            input_ent = input_ent+1
            #print(input_ent)
            #print(input_ent.shape)
            #exit()

            ###
            '''
            cordinate = np.array(torch.nonzero(input_ent))
            for x,y in cordinate:
                input_ent[x,y]=2
            '''
            ###

            #for input_id in input_ids:
                #text = tokenizer.convert_ids_to_tokens(input_id.tolist())
                #text = tokenizer.decode(input_id.tolist())
                #print(text)
            #exit()

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            input_ent = input_ent.to(device)
            ent_mask = ent_mask.to(device)
            label_ids = label_ids.to(device)


            #input_ent = input_ent%9
            #k, v = load_k_v_queryR(input_ent)
            k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)



            with torch.no_grad():
                #tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids)
                #tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, label_ids, device, input_ent_emb)
                #tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, label_ids, k.half(), v.half())
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids, k_1.half(), v_1.half(), k_2.half(), v_2.half())
                #logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask)
                #logits = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, None, device, input_ent_emb)
                #logits = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, None, k.half(), v.half())
                logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, None, k_1.half(), v_1.half(), k_2.half(), v_2.half())

            #logits = (logits_erine[i_index].cuda().half() + logits)/2 ###erine
            #i_index += 1 ###erine

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy, pred = accuracy(logits, label_ids)
            for a, b in zip(pred, label_ids):
                fgold.write("{}\n".format(b))
                fpred.write("{}\n".format(a))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy
                  }


        #with open(output_eval_file+"_comb", "w") as writer: ###erine
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()