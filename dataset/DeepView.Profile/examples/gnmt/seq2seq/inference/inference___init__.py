def __init__(self, model, tokenizer, loader, beam_size=5, len_norm_factor=
    0.6, len_norm_const=5.0, cov_penalty_factor=0.1, max_seq_len=50, cuda=
    False, print_freq=1, dataset_dir=None, save_path=None, target_bleu=None):
    self.model = model
    self.tokenizer = tokenizer
    self.loader = loader
    self.insert_target_start = [config.BOS]
    self.insert_src_start = [config.BOS]
    self.insert_src_end = [config.EOS]
    self.batch_first = model.batch_first
    self.cuda = cuda
    self.beam_size = beam_size
    self.print_freq = print_freq
    self.dataset_dir = dataset_dir
    self.target_bleu = target_bleu
    self.save_path = save_path
    self.distributed = get_world_size() > 1
    self.generator = SequenceGenerator(model=self.model, beam_size=
        beam_size, max_seq_len=max_seq_len, cuda=cuda, len_norm_factor=
        len_norm_factor, len_norm_const=len_norm_const, cov_penalty_factor=
        cov_penalty_factor)
