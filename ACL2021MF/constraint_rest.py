import numpy as np
import torch

class cbs_matrix:

    def __init__(self, vocab_size):
        self.matrix = None
        self.vocab_size = vocab_size

    def init_matrix(self, state_size):
        self.matrix = np.zeros((1, state_size, state_size, self.vocab_size), dtype=np.uint8)

    def add_connect(self, from_state, to_state, w_group):
        assert self.matrix is not None
        for w_index in w_group:
            self.matrix[0, from_state, to_state, w_index] = 1
            self.matrix[0, from_state, from_state, w_index] = 0

    def add_connect_except(self, from_state, to_state, w_group):
        excluded_group_word = [w for w in range(self.vocab_size) if w not in w_group]
        self.add_connect(from_state, to_state, excluded_group_word)

    def init_row(self, state_index):
        assert self.matrix is not None
        self.matrix[0, state_index, state_index, :] = 1

    def get_matrix(self):
        return self.matrix







def CBSConstraint(CBS_type, max_constrain_num):
    if CBS_type == 'Two':
        assert max_constrain_num <= 2
        return TwoConstraint()
    elif CBS_type == 'GBS':
        return GBSConstraint(max_constrain_num)
    else:
        raise NotImplementedError

class Constraint:

    constraint_max_length = 6
    _num_cls = {}
    _cache = {}
        

class TwoConstraint(Constraint):


                    


class GBSConstraint(Constraint):


