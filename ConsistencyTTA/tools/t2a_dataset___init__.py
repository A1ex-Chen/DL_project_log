def __init__(self, dataset_json_path, generated_path, mel_path,
    num_examples=-1, target_length=1024, sample_rate=16000):
    super().__init__()
    assert os.path.isfile(dataset_json_path
        ), f'{dataset_json_path} is not a file.'
    self.dataset = pd.read_json(dataset_json_path, lines=True)
    assert os.path.isdir(generated_path
        ), f'{generated_path} is not a directory.'
    self.generated_path = generated_path
    assert mel_path is None or os.path.isfile(mel_path
        ), f'{mel_path} is not a directory.'
    self.mel = torch.load(mel_path) if mel_path is not None else None
    self.captions = list(self.dataset['captions'])
    self.audio_paths = list(self.dataset['location'])
    self.indices = list(range(len(self.captions)))
    self.target_length = target_length
    if isinstance(sample_rate, list):
        self.sample_rate = [int(sr) for sr in sample_rate]
    else:
        self.sample_rate = int(sample_rate)
    print(f'Target sample rate: {self.sample_rate}')
    if num_examples != -1:
        self.captions = self.captions[:num_examples]
        self.audio_paths = self.audio_paths[:num_examples]
        self.indices = self.indices[:num_examples]
        if self.mel is not None:
            self.mel = self.mel[:num_examples, :, :, :]
