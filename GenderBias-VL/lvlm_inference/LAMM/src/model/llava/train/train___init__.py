def __init__(self, data_path: str, tokenizer: transformers.
    PreTrainedTokenizer, data_args: DataArguments):
    super(LazySupervisedDataset, self).__init__()
    list_data_dict = json.load(open(data_path, 'r'))
    rank0_print('Formatting inputs...Skip in lazy mode')
    self.tokenizer = tokenizer
    self.list_data_dict = list_data_dict
    self.data_args = data_args
