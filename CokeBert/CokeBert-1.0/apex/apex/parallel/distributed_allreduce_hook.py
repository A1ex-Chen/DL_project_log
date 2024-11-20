def allreduce_hook(*unused):
    if not self._disable_allreduce:
        if self.delay_allreduce or self.needs_refresh:
            if not self.delay_allreduce and self.needs_refresh:
                active_i = self.param_id_to_active_i[id(param)]
                current_type = self.param_type_to_tmp_i[param.type()]
                self.tmp_buckets[current_type].append(active_i)
                ship_tmp_bucket = False
                if self.custom_allreduce_triggers:
                    if id(param) in self.allreduce_trigger_params:
                        ship_tmp_bucket = True
                else:
                    self.tmp_numels[current_type] += param.numel()
                    if self.tmp_numels[current_type] >= self.message_size:
                        ship_tmp_bucket = True
                if ship_tmp_bucket:
                    self.active_i_buckets.append(self.tmp_buckets[current_type]
                        )
                    self.tmp_buckets[current_type] = []
                    self.tmp_numels[current_type] = 0
            if not self.callback_queued:
                Variable._execution_engine.queue_callback(allreduce_params)
                self.callback_queued = True
        else:
            if not self.callback_queued:
                Variable._execution_engine.queue_callback(
                    overlapping_backward_epilogue)
                self.callback_queued = True
            self.comm_ready_buckets(param)
