def _collate_and_pad_divisibility(tensor_list: list, div=32):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.tensor([img.shape[i] for img in
            tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    c, h, w = max_size
    pad_h = div - h % div if h % div != 0 else 0
    pad_w = div - w % div if w % div != 0 else 0
    max_size = c, h + pad_h, w + pad_w
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
    return padded_imgs
