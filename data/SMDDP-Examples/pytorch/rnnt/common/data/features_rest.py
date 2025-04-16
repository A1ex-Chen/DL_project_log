# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import librosa
import torch
import torch.nn as nn

from apex import amp


class BaseFeatures(nn.Module):
    """Base class for GPU accelerated audio preprocessing."""
    def __init__(self, optim_level):
        super(BaseFeatures, self).__init__()
        self.optim_level = optim_level

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, x):
        audio, audio_lens = x
        if self.optim_level == 1:
            with amp.disable_casts():
                return self.calculate_features(audio, audio_lens)
        else:
            return self.calculate_features(audio, audio_lens)


class SpecAugment(BaseFeatures):
    """Regularize by masking entire time steps/frequency bands.

    Implementes SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

    Args:
        freq_masks (int): number of masks for frequency bands
        min_freq (int): minimum number of frequencies in a single mask
        max_freq (int or float): maximum number of frequencies in a single mask
        time_masks (int or float): number of masks or adaptive percentage
        min_time (int): minimum number of masked time steps per mask; applies
            only if max is non-adaptive
        max_time (int or float): maximum number of masked time steps per mask,
            value 0 < 1 then denotes adaptive percentage
        noise_magnitude (float): mask with N(0, noise_magnitude * std(sample))
            noise instead of zeros to stabilize training
    """
    def __init__(self, optim_level, freq_masks=0, min_freq=0, max_freq=10, time_masks=0,
                 min_time=0, max_time=10, noise_magnitude=0):
        super(SpecAugment, self).__init__(optim_level)
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

        self.noise_magnitude = noise_magnitude

    @torch.no_grad()
    def calculate_features(self, x, x_lens):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):

            for _ in range(self.freq_masks):
                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = torch.randint(0, max(1, sh[1] - w), size=(1,))
                mask[idx, f0:f0+w] = 1

            # Adaptive time masking
            time_masks = self.time_masks
            if 0 < time_masks < 1.0:
                time_masks = int(round(x_lens[idx].item() * time_masks))

            max_time = self.max_time
            if 0 < max_time < 1.0:
                max_time = int(round(x_lens[idx].item() * max_time))

            for _ in range(time_masks):
                w = torch.randint(self.min_time, max_time + 1, size=(1,)).item()
                t0 = torch.randint(0, max(1, sh[2] - w), size=(1,))
                mask[idx, :, t0:t0+w] = 1

        if self.noise_magnitude > 0:
            mean = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            std = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            for idx in range(sh[0]):
                mean[idx, :, 0] = x[idx, :, :x_lens[idx]].mean(dim=1)
                std[idx, :, 0] = x[idx, :, :x_lens[idx]].mean(dim=1)

            std *= self.noise_magnitude
            noise = (mean + torch.randn_like(x) * std).masked_fill(~mask, 0)
        else:
            noise = 0

        return x.masked_fill(mask, 0) + noise, x_lens


class VectorizedSpecAugment(SpecAugment):
    def calculate_features(self, x, x_lens):
        assert self.noise_magnitude == 0, "noise magnitude not implemented"

        b, h, w = x.shape

        time_shape   = torch.randint(self.min_time, int(round(w * self.max_time)) + 1, [b, self.time_masks, 1], device='cuda')
        time_anchors = (torch.rand([b, self.time_masks, 1], device='cuda') * (w - time_shape)).round().int()
        time_idx     = torch.linspace(0, w-1, w, dtype=int, device='cuda')
        time_mask   = (
            (time_idx >= time_anchors) * (time_idx <= time_anchors + time_shape)
        ).any(dim=1)

        freq_shape   = torch.randint(self.min_freq, self.max_freq + 1, [b, self.freq_masks, 1], device='cuda')
        freq_anchors = (torch.rand([b, self.freq_masks, 1], device='cuda') * (h - freq_shape)).round().int()
        freq_idx     = torch.linspace(0, h-1, h, dtype=int, device='cuda')
        freq_mask   = (
            (freq_idx >= freq_anchors) * (freq_idx <= freq_anchors + freq_shape)
        ).any(dim=1)

        return x.masked_fill(time_mask.view(b,1,-1) + freq_mask.view(b,-1,1), 0), x_lens


@torch.jit.script




    @torch.no_grad()



class SpecAugment(BaseFeatures):
    """Regularize by masking entire time steps/frequency bands.

    Implementes SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

    Args:
        freq_masks (int): number of masks for frequency bands
        min_freq (int): minimum number of frequencies in a single mask
        max_freq (int or float): maximum number of frequencies in a single mask
        time_masks (int or float): number of masks or adaptive percentage
        min_time (int): minimum number of masked time steps per mask; applies
            only if max is non-adaptive
        max_time (int or float): maximum number of masked time steps per mask,
            value 0 < 1 then denotes adaptive percentage
        noise_magnitude (float): mask with N(0, noise_magnitude * std(sample))
            noise instead of zeros to stabilize training
    """

    @torch.no_grad()


class VectorizedSpecAugment(SpecAugment):


@torch.jit.script
def normalize_batch(x, x_lens, normalize_type: str):
    if normalize_type == "per_feature":
        mean = x.new_zeros(x.size(0), x.size(1))
        std = x.new_zeros(x.size(0), x.size(1))

        for i in range(x.size(0)):
            mean[i, :] = x[i, :, :x_lens[i]].mean(dim=1)
            std[i, :] = x[i, :, :x_lens[i]].std(dim=1)
        # make sure std is not zero
        return (x - mean.unsqueeze(2)) / (std.unsqueeze(2) + 1e-5)

    elif normalize_type == "all_features":
        mean = x.new_zeros(x.size(0))
        std = x.new_zeros(x.size(0))
        for i in range(x.size(0)):
            mean[i] = x[i, :, :x_lens[i]].mean()
            std[i] = x[i, :, :x_lens[i]].std()
        # make sure x_std is not zero
        return (x - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-5)
    else:
        return x


def stack_subsample_frames(x, x_lens, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]

    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()

        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:,:,:x_lens.max().item()]

    return x, x_lens

def stack_subsample_frames_no_sync(x, x_lens, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    assert stacking == subsampling

    # x is [B, H, T]
    x = x.transpose(1, 2)
    T = x.size(1)
    padded = torch.nn.functional.pad(x, (0, 0, 0, (stacking - (T % stacking)) % stacking))
    B, T, H = padded.size()
    x = padded.reshape(B, T // stacking, -1)
    x = x.transpose(1, 2)
    x_lens = (x_lens.int() + stacking - 1) // stacking
    return x, x_lens


class FilterbankFeatures(BaseFeatures):
    # For JIT, https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length",
                     "log", "normalize"]
    # torchscript: "center" removed due to a bug


    # do stft
    # TORCHSCRIPT: center removed due to bug
                          # return_complex=False)


    @torch.no_grad()

class FrameSplicing(BaseFeatures):
    __constants__ = ['frame_subsampling', 'frame_stacking']



class FillPadding(BaseFeatures):
    __constants__ = [ 'fill_value' ]


class PadAlign(BaseFeatures):
    __constants__ = [ 'pad_align_time', 'pad_align_freq', 'pad_to_max_duration', 'max_len' ]


