def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()
    return apply_to_sample(_move_to_cuda, sample)
