def __getstate__(self):
    attrs = copy.copy(self.__dict__)
    if self._backend != self.backend_enum_holder.NCCL:
        del attrs['self.reduction_stream']
        del attrs['self.reduction_event']
        return attrs
