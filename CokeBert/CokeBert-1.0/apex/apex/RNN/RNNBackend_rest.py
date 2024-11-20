import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

import math





    
#These modules always assumes batch_first
class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN
    """
        
    #collect hidden option will return all hidden/cell states from entire RNN

        

        


   
#assumes hidden_state[0] of inputRNN is output hidden state
#constructor either takes an RNNCell or list of RNN layers
class stackedRNN(nn.Module):        
    """
    stackedRNN
    """


    '''
    Returns output as hidden_state[0] Tensor([sequence steps][batch size][features])
    If collect hidden will also return Tuple(
        [n_hidden_states][sequence steps] Tensor([layer][batch size][features])
    )
    If not collect hidden will also return Tuple(
        [n_hidden_states] Tensor([layer][batch size][features])
    '''
    
        

        


class RNNCell(nn.Module):
    """ 
    RNNCell 
    gate_multiplier is related to the architecture you're working with
    For LSTM-like it will be 4 and GRU-like will be 3.
    Always assumes input is NOT batch_first.
    Output size that's not hidden size will use output projection
    Hidden_states is number of hidden states that are needed for cell
    if one will go directly to cell as tensor, if more will go as list
    """


    
    #Use xavier where we can (weights), otherwise use uniform (bias)
    '''
    Xavier reset:
    def reset_parameters(self, gain=1):
        stdv = 1.0 / math.sqrt(self.gate_size)

        for param in self.parameters():
            if (param.dim() > 1):
                torch.nn.init.xavier_normal(param, gain)
            else:
                param.data.uniform_(-stdv, stdv)
    '''
            
        

        