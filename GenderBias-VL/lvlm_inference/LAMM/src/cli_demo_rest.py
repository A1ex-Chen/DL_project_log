import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from copy import deepcopy
import time
from model.LAMM import LAMMPEFTModel
import torch
import json
import argparse
from model.LAMM.conversations import conv_templates
from tqdm import tqdm

INPUT_KEYS = ['image_path', 'images', 'pcl_path']
SYS_MSG = """
You are an AI visual assistant that can analyze a single point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. 
As an AI assistant, you are performing a visual question answering task, and your goal is to generate natural language answers that accurately solve the question. In order to generate accurate answers to questions about visual content, you must be able to understand the content of point cloud, understand the meaning of questions, perform complex reasoning processes, and give out determined results.
"""



    





    
    
if __name__ == '__main__':
    args = parse_args()
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    main(args)