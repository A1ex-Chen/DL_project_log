def __init__(self, model, rnnt_config, batch_size, max_feat_len,
    max_txt_len, num_cg):
    self.model = model
    self.rnnt_config = rnnt_config
    self.batch_size = batch_size
    self.cg_stream = torch.cuda.Stream()
    self.encode_stream = torch.cuda.Stream()
    self.predict_stream = torch.cuda.Stream()
    self.max_feat_len = max_feat_len
    self.max_txt_len = max_txt_len
    self.num_cg = num_cg
