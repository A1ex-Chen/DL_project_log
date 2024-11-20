def __del__(self):
    if self.bn_group > 1:
        bnp.close_remote_data(self.pair_handle)
        if self.bn_group > 2:
            bnp.close_remote_data(self.pair_handle2)
            if self.bn_group > 4:
                bnp.close_remote_data(self.pair_handle3)
