def after_step(self):
    if self._runs > self._max_runs:
        return
    if (self.trainer.iter + 1
        ) % self._period == 0 or self.trainer.iter == self.trainer.max_iter - 1:
        if torch.cuda.is_available():
            max_reserved_mb = torch.cuda.max_memory_reserved(
                ) / 1024.0 / 1024.0
            reserved_mb = torch.cuda.memory_reserved() / 1024.0 / 1024.0
            max_allocated_mb = torch.cuda.max_memory_allocated(
                ) / 1024.0 / 1024.0
            allocated_mb = torch.cuda.memory_allocated() / 1024.0 / 1024.0
            self._logger.info(
                ' iter: {}  max_reserved_mem: {:.0f}MB  reserved_mem: {:.0f}MB  max_allocated_mem: {:.0f}MB  allocated_mem: {:.0f}MB '
                .format(self.trainer.iter, max_reserved_mb, reserved_mb,
                max_allocated_mb, allocated_mb))
            self._runs += 1
            if self._runs == self._max_runs:
                mem_summary = torch.cuda.memory_summary()
                self._logger.info('\n' + mem_summary)
            torch.cuda.reset_peak_memory_stats()
