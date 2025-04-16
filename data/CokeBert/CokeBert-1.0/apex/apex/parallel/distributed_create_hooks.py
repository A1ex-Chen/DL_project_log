def create_hooks(self):

    def allreduce_params():
        if not self.delay_allreduce:
            if self.needs_refresh:
                self.sync_bucket_structure()
                self.needs_refresh = False
        self.allreduce_fallback()

    def overlapping_backward_epilogue():
        self.reduction_stream.record_event(self.reduction_event)
        torch.cuda.current_stream().wait_event(self.reduction_event)
        if self.next_bucket != self.num_buckets:
            raise RuntimeError(
                'In epilogue, next_bucket ({}) != num_buckets ({}).  '.
                format(self.next_bucket, self.num_buckets),
                'This probably indicates some buckets were not allreduced.')
        for actual, expected in zip(self.buckets_ready_size, self.bucket_sizes
            ):
            if actual != expected:
                raise RuntimeError('Some param buckets were not allreduced.')
    self.grad_accs = []
    for param in self.module.parameters():
        if param.requires_grad:

            def wrapper(param):
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def allreduce_hook(*unused):
                    if not self._disable_allreduce:
                        if self.delay_allreduce or self.needs_refresh:
                            if not self.delay_allreduce and self.needs_refresh:
                                active_i = self.param_id_to_active_i[id(param)]
                                current_type = self.param_type_to_tmp_i[param
                                    .type()]
                                self.tmp_buckets[current_type].append(active_i)
                                ship_tmp_bucket = False
                                if self.custom_allreduce_triggers:
                                    if id(param
                                        ) in self.allreduce_trigger_params:
                                        ship_tmp_bucket = True
                                else:
                                    self.tmp_numels[current_type
                                        ] += param.numel()
                                    if self.tmp_numels[current_type
                                        ] >= self.message_size:
                                        ship_tmp_bucket = True
                                if ship_tmp_bucket:
                                    self.active_i_buckets.append(self.
                                        tmp_buckets[current_type])
                                    self.tmp_buckets[current_type] = []
                                    self.tmp_numels[current_type] = 0
                            if not self.callback_queued:
                                Variable._execution_engine.queue_callback(
                                    allreduce_params)
                                self.callback_queued = True
                        else:
                            if not self.callback_queued:
                                Variable._execution_engine.queue_callback(
                                    overlapping_backward_epilogue)
                                self.callback_queued = True
                            self.comm_ready_buckets(param)
                grad_acc.register_hook(allreduce_hook)
                self.grad_accs.append(grad_acc)
            wrapper(param)
