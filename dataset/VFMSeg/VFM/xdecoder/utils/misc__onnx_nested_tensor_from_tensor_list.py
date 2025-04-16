@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]
    ) ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in
            tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0,
            padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding
            [1]), 'constant', 1)
        padded_masks.append(padded_mask.to(torch.bool))
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)
