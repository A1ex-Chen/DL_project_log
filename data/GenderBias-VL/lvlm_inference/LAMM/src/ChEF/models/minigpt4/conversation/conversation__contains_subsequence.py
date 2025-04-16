def _contains_subsequence(self, large_tensor, small_tensor):
    len_small = len(small_tensor)
    for i in range(0, len(large_tensor) - len_small + 1):
        flag = torch.all(small_tensor == large_tensor[i:i + len_small]).item()
        if flag:
            return True
    return False
