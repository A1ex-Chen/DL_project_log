def allreduce_bucket(self, bucket):
    tensor = flatten(bucket)
    tensor_to_allreduce = tensor
    if self.allreduce_always_fp32:
        tensor_to_allreduce = tensor.float()
    if self.gradient_predivide_factor != 1.0:
        tensor_to_allreduce.mul_(1.0 / self.gradient_predivide_factor)
    dist.all_reduce(tensor_to_allreduce)
    if self.gradient_average:
        if self.gradient_predivide_factor != self.world_size:
            tensor_to_allreduce.mul_(self.gradient_predivide_factor / self.
                world_size)
    if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
        tensor.copy_(tensor_to_allreduce)
    return tensor
