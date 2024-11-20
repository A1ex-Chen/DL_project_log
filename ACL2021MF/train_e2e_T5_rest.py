from dataset.vocabulary import T5CopyVocabulary
from dataset.e2e_dataset import E2EDataset, get_data_loader
import argparse
import torch
import torch.nn as nn
from config import Config
import numpy as np
from transformers import T5Tokenizer
from checkpointing import CheckpointManager
from t5 import get_lm_representation 
import utils
from tqdm import tqdm
import math
import os, sys
from speaksee import evaluation
import random
from dataset.pymteval import BLEUScore, NISTScore
from dataset.diversity import distinct_n
from constraint import CBSConstraint
import json

parser = argparse.ArgumentParser("Train a CommonGen T5")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--constraint-vocab",
    default=None,
    help="Path to load constraint vocab",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')



if __name__ == "__main__":
    _A = parser.parse_args()

    _C = Config(_A.config, _A.config_override)

    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained(_C.lm_type, cache_dir='.')
    copy_vocab = T5CopyVocabulary(_C.copy_vocab_path, tokenizer, sep='@')
    lm = get_lm_representation(_C, tokenizer, copy_vocab)
    model = lm['t5']
    model = model.to(device)
    _C.vocab_size = model.config.vocab_size

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)

    if len(_C.decode_constrain) > 0:
        decode_constraint = CBSConstraint(_C.decode_constrain, 2)
    else:
        decode_constraint = None

    if _A.train:
        train_data = E2EDataset(_C, _C.train_path, tokenizer, copy_vocab, is_training=True)
        train_data_loader = get_data_loader(train_data, _C.batch_size)
        train_iter = iter(train_data_loader)

    dev_data = E2EDataset(_C, _C.dev_path if (_A.validation or _A.train) else _C.test_path, tokenizer, copy_vocab)
    dev_data_loader = get_data_loader(dev_data, _C.batch_size)

    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    if _A.validation or _A.test:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

        run_eval(_C, model, dev_data_loader, tokenizer, copy_vocab, device, decode_constraint=decode_constraint, constraint_vocab=_A.constraint_vocab, output_path=_A.output_path)


    if _A.train:
        _C.num_training_steps = len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps
        epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

        checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")
        optimizer = utils.build_optimizer(_C, model)

        os.makedirs(_A.serialization_dir, exist_ok=True)
        _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

        eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
        total_step = 0

        for epoch in range(epoch_num):
            print('EPOCH %d / %d' % (epoch + 1, epoch_num))
            run_step = eval_every if total_step + eval_every < len(train_iter) * _C.max_epoch else  len(train_iter) * _C.max_epoch - total_step
            model.train()

            with tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout) as pbar:
                for step in range(run_step):
                    try:
                        batch = next(train_iter)
                    except:
                        train_iter = iter(train_data_loader)
                        batch = next(train_iter)
                   
                    for n in batch:
                        if n not in ['gt', 'gt_mr', 'ins_id']:
                            batch[n] = batch[n].to(device)
                    # optimizer.zero_grad()
                    total_step += 1
                    outputs = model(
                        input_ids=batch['encoder_input_ids'], 
                        attention_mask=batch['encoder_mask'],
                        decoder_mention_flag=batch['mention_flag'],
                        decoder_cls_on_input=batch['encoder_cls'],
                        labels=batch['cap_decoder_input_ids']
                    )
                    loss = outputs.loss
                    loss = loss / _C.gradient_accumulation_steps
                    loss.backward()

                    if _C.grad_clip_value > 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), _C.grad_clip_value)
                    if (step + 1) % _C.gradient_accumulation_steps == 0:
                        optimizer.step()
                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                        pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
                        pbar.update(1)
                        optimizer.zero_grad()

            eval_result = run_eval(_C, model, dev_data_loader, tokenizer, copy_vocab, device)
            checkpoint_manager.step(eval_result["CIDEr"]["entire"])