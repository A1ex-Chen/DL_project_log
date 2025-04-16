# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.log_uniform_sampler import LogUniformSampler
from utils.log_uniform_sampler import sample_logits
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


@torch.jit.script


class PositionalEmbedding(nn.Module):



class PositionwiseFF(nn.Module):



class MultiHeadAttn(nn.Module):



class RelMultiHeadAttn(nn.Module):






class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):



class RelLearnableMultiHeadAttn(RelMultiHeadAttn):



class DecoderLayer(nn.Module):



class RelLearnableDecoderLayer(nn.Module):



class RelPartialLearnableDecoderLayer(nn.Module):



class AdaptiveEmbedding(nn.Module):



class MemTransformerLM(nn.Module):









if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner,
                                     args.dropout, dropatt=args.dropout,
                                     tie_weight=True, d_embed=d_embed,
                                     div_val=div_val, tie_projs=tie_projs,
                                     pre_lnorm=True, tgt_len=tgt_len,
                                     ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0,
                                     dtype=None).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = None
            for idx, (inp, tgt, seqlen, _) in enumerate(diter):
                print('batch {}'.format(idx))
                _, mems = model(inp, tgt, mems)