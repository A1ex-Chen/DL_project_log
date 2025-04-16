def store_flos(self):
    if self._total_flos is not None:
        if self.args.local_rank != -1:
            self.state.total_flos = distributed_broadcast_scalars([self.
                _total_flos]).sum().item()
        else:
            self.state.total_flos = self._total_flos
