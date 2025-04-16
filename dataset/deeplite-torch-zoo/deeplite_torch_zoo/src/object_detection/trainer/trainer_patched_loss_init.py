def patched_loss_init(obj, model):
    device = next(model.parameters()).device
    h = model.args
    m = model.detection if hasattr(model, 'detection') else model.model[-1]
    obj.bce = nn.BCEWithLogitsLoss(reduction='none')
    obj.hyp = h
    obj.stride = m.stride
    obj.nc = m.nc
    obj.no = m.no
    obj.reg_max = m.reg_max
    obj.device = device
    obj.use_dfl = m.reg_max > 1
    obj.assigner = TaskAlignedAssigner(topk=10, num_classes=obj.nc, alpha=
        0.5, beta=6.0)
    obj.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=obj.use_dfl).to(device)
    obj.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
