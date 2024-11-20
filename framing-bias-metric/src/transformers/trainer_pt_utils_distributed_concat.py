def distributed_concat(tensor: 'torch.Tensor', num_total_examples: Optional
    [int]=None) ->torch.Tensor:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for
                t in tensor)
        output_tensors = [tensor.clone() for _ in range(torch.distributed.
            get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError('Not currently using distributed training')
