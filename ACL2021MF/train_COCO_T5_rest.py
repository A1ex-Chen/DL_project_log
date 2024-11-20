import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import sys
import json
import utils
import random

from config import Config
from dataset.coco_dataset import COCODataset, get_data_loader
from checkpointing import CheckpointManager
from dataset import evaluation
from t5 import get_lm_representation 
from dataset.vocabulary import T5CopyVocabulary
from transformers import T5Tokenizer
from dataset.EvalAI import NocapsEvaluator
from constraint import CBSConstraint
from dataset.diversity import distinct_n

parser = argparse.ArgumentParser("Train a Transformer Captioner with RL")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--eval-split", help="Path to the evaluation split"
)
parser.add_argument(
    "--in-memory", action="store_true", help="Whether to load image features in memory."
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
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
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
parser.add_argument(
    "--cbs-class-path",
    default=None,
    help="Path to a (non-existent) directory for CBS class path.",
)
parser.add_argument(
    "--novel-constraint-path",
    default=None,
    help="Path to novel constraints",
)
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')
parser.add_argument('--port',  type=int, default=8083, help='port for server to run')
parser.add_argument('--host',  type=str, default='localhost', help='host for server to run')


if __name__ == "__main__":

    _A = parser.parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    _C = Config(_A.config, _A.config_override)

    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = T5Tokenizer.from_pretrained(_C.lm_type, cache_dir='.')
    copy_vocab = T5CopyVocabulary(_C.copy_vocab_path, tokenizer)
    lm = get_lm_representation(_C, tokenizer, copy_vocab)
    model = lm['t5']
    model = model.to(device)
    attachable_index = lm['attachable_index']
    _C.vocab_size = model.config.vocab_size

    if len(_C.decode_constrain) > 0:
        decode_constraint = CBSConstraint(_C.decode_constrain, 2)
    else:
        decode_constraint = None

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)

    if _C.use_copy_obj:
        train_copy_obj_h5_path = _C.train_copy_obj_h5_path
        dev_copy_obj_h5_path = _C.dev_copy_obj_h5_path
        test_copy_obj_h5_path = _C.test_copy_obj_h5_path
    else:
        train_copy_obj_h5_path, dev_copy_obj_h5_path, test_copy_obj_h5_path = None, None, None

    if _A.train:
        train_data = COCODataset(_C, _C.train_obj_h5_path, tokenizer, copy_vocab, attachable_index, caption_path=_C.train_path, copy_h5_path=train_copy_obj_h5_path, in_memory=_A.in_memory, is_training=True)
        train_data_loader = get_data_loader(_C, train_data)
        train_iter = iter(train_data_loader)

    if not _A.test:
        val_data = COCODataset(_C, _C.dev_obj_h5_path, tokenizer, copy_vocab, attachable_index, caption_path=_C.dev_path, copy_h5_path=dev_copy_obj_h5_path, is_training=False, in_memory=_A.in_memory, cbs_class_path=_A.cbs_class_path)
    else:
        val_data = COCODataset(_C, _C.test_obj_h5_path, tokenizer, copy_vocab, attachable_index, caption_path=_C.test_path, copy_h5_path=test_copy_obj_h5_path, is_training=False, in_memory=_A.in_memory, cbs_class_path=_A.cbs_class_path)
    val_data_loader = get_data_loader(_C, val_data)

    if _A.start_from_checkpoint is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

    if _A.validation or _A.test:
        assert _A.start_from_checkpoint is not None, "evaluation must come along with pre-trained model"
        run_eval(_C, model, val_data_loader, tokenizer, copy_vocab, device, output_path=_A.output_path, test=_A.test, full_eval=True, decode_constraint=decode_constraint, novel_constraint_path=_A.novel_constraint_path)
    
    if _A.train:
        model.train()
        _C.num_training_steps = len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps
        epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)
        
        optimizer = utils.build_optimizer(_C, model)
        checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")
        eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps

        total_step = 0

        print(_C)
        for arg in vars(_A):
            print("{:<20}: {}".format(arg, getattr(_A, arg)))

        os.makedirs(_A.serialization_dir, exist_ok=True)
        _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

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

                    if torch.cuda.is_available():
                        for n in batch:
                            if n in ['gt', 'image_ids']: continue
                            batch[n] = batch[n].cuda()

                    total_step += 1

                    # optimizer.zero_grad()

                    outputs = model(
                        input_ids=batch['encoder_input_ids'], 
                        attention_mask=batch['encoder_mask'],
                        encoder_img_mask=batch['encoder_img_mask'],
                        encoder_obj_feature=batch['encoder_obj_feature'],
                        encoder_obj_box=batch['encoder_obj_box'],
                        encoder_relative_pos_index=batch['encoder_rel_position'],
                        decoder_mention_flag=batch['mention_flag'],
                        decoder_cls_on_input=batch['encoder_cls'],
                        labels=batch['cap_decoder_input_ids']
                    )
                    #training
                    loss = outputs.loss
                    loss = loss / _C.gradient_accumulation_steps
                    loss.backward()
                    if _C.grad_clip_value != 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), _C.grad_clip_value)
                    if (step + 1) % _C.gradient_accumulation_steps == 0:
                        optimizer.step()
                        pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
                        pbar.update(1)
                        optimizer.zero_grad()

            eval_result = run_eval(_C, model, val_data_loader, tokenizer, copy_vocab, device, output_path=_A.output_path)
            checkpoint_manager.step(eval_result["CIDEr"]["entire"])
