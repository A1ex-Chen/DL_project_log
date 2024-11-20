def save_one_txt(self, predn, save_conf, shape, file):
    """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
    gn = torch.tensor(shape)[[1, 0]]
    for *xywh, conf, cls, angle in predn.tolist():
        xywha = torch.tensor([*xywh, angle]).view(1, 5)
        xyxyxyxy = (ops.xywhr2xyxyxyxy(xywha) / gn).view(-1).tolist()
        line = (cls, *xyxyxyxy, conf) if save_conf else (cls, *xyxyxyxy)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
