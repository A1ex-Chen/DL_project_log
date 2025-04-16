def nms_forward_function(op: Operation, values: List[torch.Tensor], **kwards
    ) ->List[torch.Tensor]:
    return torch.zeros([1, 1], dtype=torch.int32).cuda(), torch.zeros([1, 
        100, 4], dtype=torch.float32).cuda(), torch.zeros([1, 100], dtype=
        torch.float32).cuda(), torch.zeros([1, 100], dtype=torch.int32).cuda()
