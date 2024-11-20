def __init__(self, train_dataset, test_dataset,
    sentence_transformers_model_name: Optional[str]='all-mpnet-base-v2',
    seed: Optional[int]=43, ice_num: Optional[int]=1, tokenizer_name:
    Optional[str]='gpt2-xl', batch_size: Optional[int]=1, **kwargs) ->None:
    super().__init__(train_dataset, test_dataset, seed, ice_num)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.batch_size = batch_size
    self.tokenizer_name = tokenizer_name
    gen_datalist = self.get_corpus_from_dataset(test_dataset)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    self.tokenizer.padding_side = 'right'
    self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer
        )
    co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=
        self.device)
    self.dataloader = DataLoader(self.encode_dataset, batch_size=self.
        batch_size, collate_fn=co)
    self.model = SentenceTransformer(sentence_transformers_model_name)
    self.model = self.model.to(self.device)
    self.model.eval()
    self.index = self.create_index()
