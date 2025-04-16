def __init__(self, num_features, fuse_relu=False, bn_group=1,
    max_cta_per_sm=2, cta_launch_margin=12, multi_stream=False):
    super(BatchNorm2d_NHWC, self).__init__(num_features)
    self.fuse_relu = fuse_relu
    self.multi_stream = multi_stream
    self.minibatch_mean = torch.cuda.FloatTensor(num_features)
    self.minibatch_riv = torch.cuda.FloatTensor(num_features)
    self.bn_group = bn_group
    self.max_cta_per_sm = max_cta_per_sm
    self.cta_launch_margin = cta_launch_margin
    self.my_data = None
    self.pair_data = None
    self.pair_data2 = None
    self.pair_data3 = None
    self.local_rank = 0
    self.magic = torch.IntTensor([0])
    assert max_cta_per_sm > 0
    self.fwd_occupancy = min(bnp.bn_fwd_nhwc_occupancy(), max_cta_per_sm)
    self.bwd_occupancy = min(bnp.bn_bwd_nhwc_occupancy(), max_cta_per_sm)
    self.addrelu_fwd_occupancy = min(bnp.bn_addrelu_fwd_nhwc_occupancy(),
        max_cta_per_sm)
    self.addrelu_bwd_occupancy = min(bnp.bn_addrelu_bwd_nhwc_occupancy(),
        max_cta_per_sm)
    mp_count = torch.cuda.get_device_properties(None).multi_processor_count
    self.fwd_grid_dim_x = max(mp_count * self.fwd_occupancy -
        cta_launch_margin, 1)
    self.bwd_grid_dim_x = max(mp_count * self.bwd_occupancy -
        cta_launch_margin, 1)
    self.addrelu_fwd_grid_dim_x = max(mp_count * self.addrelu_fwd_occupancy -
        cta_launch_margin, 1)
    self.addrelu_bwd_grid_dim_x = max(mp_count * self.addrelu_bwd_occupancy -
        cta_launch_margin, 1)
    self.grid_dim_y = (num_features + 63) // 64
    self.ret_cta = torch.cuda.ByteTensor(8192).fill_(0)
    if bn_group > 1:
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        assert world_size >= bn_group
        assert world_size % bn_group == 0
        bn_sync_steps = 1
        if bn_group == 4:
            bn_sync_steps = 2
        if bn_group == 8:
            bn_sync_steps = 3
        self.ipc_buffer = torch.cuda.ByteTensor(bnp.get_buffer_size(
            bn_sync_steps))
        self.my_data = bnp.get_data_ptr(self.ipc_buffer)
        self.storage = self.ipc_buffer.storage()
        self.share_cuda = self.storage._share_cuda_()
        internal_cuda_mem = self.share_cuda
        my_handle = torch.cuda.ByteTensor(np.frombuffer(internal_cuda_mem[1
            ], dtype=np.uint8))
        my_offset = torch.cuda.IntTensor([internal_cuda_mem[3]])
        handles_all = torch.empty(world_size, my_handle.size(0), dtype=
            my_handle.dtype, device=my_handle.device)
        handles_l = list(handles_all.unbind(0))
        torch.distributed.all_gather(handles_l, my_handle)
        offsets_all = torch.empty(world_size, my_offset.size(0), dtype=
            my_offset.dtype, device=my_offset.device)
        offsets_l = list(offsets_all.unbind(0))
        torch.distributed.all_gather(offsets_l, my_offset)
        self.pair_handle = handles_l[local_rank ^ 1].cpu().contiguous()
        pair_offset = offsets_l[local_rank ^ 1].cpu()
        self.pair_data = bnp.get_remote_data_ptr(self.pair_handle, pair_offset)
        if bn_group > 2:
            self.pair_handle2 = handles_l[local_rank ^ 2].cpu().contiguous()
            pair_offset2 = offsets_l[local_rank ^ 2].cpu()
            self.pair_data2 = bnp.get_remote_data_ptr(self.pair_handle2,
                pair_offset2)
        if bn_group > 4:
            self.pair_handle3 = handles_l[local_rank ^ 4].cpu().contiguous()
            pair_offset3 = offsets_l[local_rank ^ 4].cpu()
            self.pair_data3 = bnp.get_remote_data_ptr(self.pair_handle3,
                pair_offset3)
        self.magic = torch.IntTensor([2])
        self.local_rank = local_rank
