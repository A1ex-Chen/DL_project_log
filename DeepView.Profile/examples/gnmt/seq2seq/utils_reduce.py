def reduce(self, op):
    """
        Reduces average value over all workers.

        :param op: 'sum' or 'mean', reduction operator
        """
    if op not in ('sum', 'mean'):
        raise NotImplementedError
    distributed = get_world_size() > 1
    if distributed:
        if hasattr(dist, 'get_backend'):
            _backend = dist.get_backend()
            if hasattr(dist, 'DistBackend'):
                backend_enum_holder = dist.DistBackend
            else:
                backend_enum_holder = dist.Backend
        else:
            _backend = dist._backend
            backend_enum_holder = dist.dist_backend
        cuda = _backend == backend_enum_holder.NCCL
        if cuda:
            avg = torch.cuda.FloatTensor([self.avg])
            _sum = torch.cuda.FloatTensor([self.sum])
        else:
            avg = torch.FloatTensor([self.avg])
            _sum = torch.FloatTensor([self.sum])
        _reduce_op = dist.reduce_op if hasattr(dist, 'reduce_op'
            ) else dist.ReduceOp
        dist.all_reduce(avg, op=_reduce_op.SUM)
        dist.all_reduce(_sum, op=_reduce_op.SUM)
        self.avg = avg.item()
        self.sum = _sum.item()
        if op == 'mean':
            self.avg /= get_world_size()
            self.sum /= get_world_size()
