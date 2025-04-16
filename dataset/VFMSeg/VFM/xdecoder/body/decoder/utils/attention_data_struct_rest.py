# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

predict_name_matcher = {"predictions_class": ["pred_logits"], 
                        "predictions_mask":["pred_masks", "pred_gmasks", "pred_smasks"], 
                        "predictions_caption":["pred_captions", "pred_gtexts"], 
                        "predictions_maskemb":["pred_maskembs", "pred_smaskembs"], 
                        "predictions_pos_spatial":["pred_pspatials"],
                        "predictions_neg_spatial":["pred_nspatials"],
                        "predictions_pos_visual":["pred_pvisuals"],
                        "predictions_neg_visual":["pred_nvisuals"]}

predict_index_matcher = {"predictions_class": ["queries_object"], 
                         "predictions_mask":["queries_object", "queries_grounding", "queries_spatial"], 
                         "predictions_caption": ["queries_object", "queries_grounding"], 
                         "predictions_maskemb":["queries_object", "queries_spatial"], 
                         "predictions_pos_spatial":["all"],
                         "predictions_neg_spatial":["all"],
                         "predictions_pos_visual":["all"],
                         "predictions_neg_visual":["all"]}

class Variable(object):
    '''
    Store dataset variable for attention
    output: embedding that accumuates during cross/self attention
    pos: positional embedding that is fixed during cross/self attention
    name: name of the variable
    type: type of the variable, e.g. queries, tokens
    attn_mask: attention mask for corss attention
    masking: masking for padding
    '''
    

class AttentionDataStruct(nn.Module):
    '''
    Store dataset structure for cross/self attention
    task_switch: switch for different tasks

    p_attn_variables: prototype of variables that is used in cross/self attention
    p_self_attn: prototype of variables that is used in self attention
    p_cross_attn: prototype of variables that is used in cross attention
    p_iter: prototype of iteration for different queries
    p_masking: prototype of masking for different tokens
    p_duplication: prototype of duplication for different quries
    '''



    
    

    



