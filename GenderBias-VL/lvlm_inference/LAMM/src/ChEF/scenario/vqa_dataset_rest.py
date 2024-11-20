import os
import json
from torch.utils.data import Dataset


OPTION=['A','B','C','D','E','F','G','H']
OPTION_MAP = {'natural':[['1','2','3','4','5','6','7','8'],
                          ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'],
                          ['first','second', 'third', 'fourth', 'fifth','sixth'],
                          ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)'],
                         ['α','β','γ','δ','ε','ζ','η','θ']],
             'neutral':[
                 ["Smith", "Johnson", "Williams", "Jones", "Brown","Davis", "Miller", "Wilson"],
                 ["foo", "dog", "hip", "oh",'cat','lake','river','joy'],
                 ['~','@','#','$', '%','^','&','*'],
                 
                ]
}


class ScienceQADataset(Dataset):
    """Example:
        data['question'] = "Question: What is the name of the colony shown?\nOptions: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont\n"
        data['options'] = ['(A', '(B', '(C', '(D']
    """
    task_name = 'VQA'
    dataset_name = 'ScienceQA'



    
    

if __name__ == '__main__':
    scienceqadata = ScienceQADataset(base_data_path='../../../data/LAMM/2D_Benchmark', ppl=True, generative=True)
    import ipdb;ipdb.set_trace()