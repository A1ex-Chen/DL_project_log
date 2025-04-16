@property
@tf_required
def n_gpu(self) ->int:
    if self.cuda:
        return len(self.gpu_list)
    return 0
