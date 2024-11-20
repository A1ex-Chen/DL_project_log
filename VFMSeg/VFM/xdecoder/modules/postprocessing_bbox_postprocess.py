def bbox_postprocess(result, input_size, img_size, output_height, output_width
    ):
    """
    result: [xc,yc,w,h] range [0,1] to [x1,y1,x2,y2] range [0,w], [0,h]
    """
    if result is None:
        return None
    scale = torch.tensor([input_size[1], input_size[0], input_size[1],
        input_size[0]])[None, :].to(result.device)
    result = result.sigmoid() * scale
    x1, y1, x2, y2 = result[:, 0] - result[:, 2] / 2, result[:, 1] - result[
        :, 3] / 2, result[:, 0] + result[:, 2] / 2, result[:, 1] + result[:, 3
        ] / 2
    h, w = img_size
    x1 = x1.clamp(min=0, max=w)
    y1 = y1.clamp(min=0, max=h)
    x2 = x2.clamp(min=0, max=w)
    y2 = y2.clamp(min=0, max=h)
    box = torch.stack([x1, y1, x2, y2]).permute(1, 0)
    scale = torch.tensor([output_width / w, output_height / h, output_width /
        w, output_height / h])[None, :].to(result.device)
    box = box * scale
    return box
