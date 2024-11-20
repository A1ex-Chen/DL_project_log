def forward(self, *inputs, **kwargs):
    result = self.module(*inputs, **kwargs)
    if not self._disable_allreduce:
        if not self.delay_allreduce:
            param_list = [param for param in self.module.parameters() if
                param.requires_grad]
            if not self.active_params or len(param_list) != len(self.
                active_params) or any([(param1 is not param2) for param1,
                param2 in zip(param_list, self.active_params)]):
                self.needs_refresh = True
            if self.needs_refresh:
                self.active_i_buckets = []
                self.buckets = []
                self.tmp_buckets = [[], [], []]
                self.tmp_numels = [0, 0, 0]
                self.bucket_sizes = []
                self.param_id_to_active_i = {id(param): i for i, param in
                    enumerate(param_list)}
                self.param_id_to_bucket = {}
            else:
                self.buckets = [[None for _ in range(self.bucket_sizes[i])] for
                    i in range(self.num_buckets)]
                self.buckets_ready_size = [(0) for i in range(self.num_buckets)
                    ]
                if self.retain_allreduce_buffers:
                    self.allreduce_buffers = [None for _ in range(self.
                        num_buckets)]
                self.next_bucket = 0
                self.ready_buckets_not_reduced = set()
            self.active_params = param_list
        self.callback_queued = False
    return result
