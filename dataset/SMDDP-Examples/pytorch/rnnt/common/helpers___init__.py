def __init__(self, feat_proc, dist_lamb, apex_transducer_joint,
    batch_split_factor, cfg):
    self.feat_proc = feat_proc
    self.dist_lamb = dist_lamb
    self.apex_transducer_joint = apex_transducer_joint
    self.enc_stack_time_factor = cfg['rnnt']['enc_stack_time_factor']
    self.window_stride = cfg['input_val']['filterbank_features'][
        'window_stride']
    self.frame_subsampling = cfg['input_val']['frame_splicing'][
        'frame_subsampling']
    self.batch_split_factor = batch_split_factor
    self.list_packed_batch_cpu = [torch.tensor(0, dtype=torch.int64, device
        ='cpu').pin_memory() for i in range(self.batch_split_factor)]
