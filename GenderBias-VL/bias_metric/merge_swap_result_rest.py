from collections import OrderedDict
import copy
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

exp_dir = "./bias_merge_swap_test"
os.makedirs(exp_dir, exist_ok=True)








if __name__ == '__main__':
    result_base_dir = "./bias_results/"

    eval_datasets = ["VLbias", "VLbias_swap", "Vbias", "Vbias_swap", "Lbias", "Lbias_swap"]
    

    model_names = ["LLaVA1.5", "LLaVA1.5-13b", "LLaVA1.6-13b", "MiniGPT-4-v2", "mPLUG-Owl2",
               "LLaMA-Adapter-v2", "InstructBLIP", "Otter", "LAMM", 
               "Kosmos2", "QwenVL", 
               "InternLMXComposer", 
               "Shikra", "LLaVARLHF", "RLHFV"] 
    
    
    for i in range(0, len(eval_datasets), 2):
        all_data = []
        for model_name in model_names:
            model_dir = os.path.join(result_base_dir, model_name)
            target_name = "occ_bias_pair_probablity_difference.csv"
            sub_file = os.path.join(model_dir, eval_datasets[i], target_name)
            sub_file_swap = os.path.join(model_dir, eval_datasets[i+1], target_name)
            merge(sub_file, sub_file_swap, model_name, eval_datasets[i])

            target_name = "occ_bias_pair_outcome_difference.csv"
            sub_file = os.path.join(model_dir, eval_datasets[i], target_name)
            sub_file_swap = os.path.join(model_dir, eval_datasets[i+1], target_name)
            merge_outcome(sub_file, sub_file_swap, model_name, eval_datasets[i])
