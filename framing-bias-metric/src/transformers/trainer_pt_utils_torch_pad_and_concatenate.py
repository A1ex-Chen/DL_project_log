def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1],
        tensor2.shape[1])) + tensor1.shape[2:]
    result = tensor1.new_full(new_shape, padding_index)
    result[:tensor1.shape[0], :tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, :tensor2.shape[1]] = tensor2
    return result
