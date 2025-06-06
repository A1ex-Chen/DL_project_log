# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import os

import numpy as np






@contextlib.contextmanager







    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception
            if any elements are filtered. Default: ``False``
    """

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)

    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception((
                'Size of sample #{} is invalid (={}) since max_positions={}, '
                'skip this example with --skip-invalid-size-inputs-valid-test'
            ).format(ignored[0], size_fn(ignored[0]), max_positions))
        yield idx

    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch.
            Default: ``None``
        max_sentences (int, optional): max number of sentences in each
            batch. Default: ``None``
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N. Default: ``1``
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf')
    bsz_mult = required_batch_size_multiple

    batch = []


    sample_len = 0
    sample_lens = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch
