def slice_tokens_for_mlm(A, indx, num_elem=2):
    all_indx = indx[:, None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]
