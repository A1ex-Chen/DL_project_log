def forward(self, x, z=None):
    if z is not None:
        assert self.fuse_relu == True
        return bn_addrelu_NHWC_impl.apply(x, z, self.weight, self.bias,
            self.running_mean, self.running_var, self.minibatch_mean, self.
            minibatch_riv, self.grid_dim_y, self.ret_cta, self.momentum,
            self.eps, self.training, self.bn_group, self.my_data, self.
            pair_data, self.magic, self.pair_data2, self.pair_data3, self.
            addrelu_fwd_occupancy, self.addrelu_fwd_grid_dim_x, self.
            addrelu_bwd_occupancy, self.addrelu_bwd_grid_dim_x, self.
            multi_stream)
    else:
        return bn_NHWC_impl.apply(x, self.weight, self.bias, self.
            running_mean, self.running_var, self.minibatch_mean, self.
            minibatch_riv, self.ret_cta, self.momentum, self.eps, self.
            fuse_relu, self.training, self.bn_group, self.my_data, self.
            pair_data, self.magic, self.pair_data2, self.pair_data3, self.
            fwd_occupancy, self.fwd_grid_dim_x, self.bwd_occupancy, self.
            bwd_grid_dim_x, self.multi_stream)
