@torch.no_grad()
def run(data, weights=None, batch_size=32, img_size=640, task='val', device
    ='', save_dir='', name=''):
    """
    TensorRT models's evaluation process.
    """
    assert task == 'val', f'task type can only be val, however you set it to {task}'
    save_dir = str(increment_name(osp.join(save_dir, name)))
    os.makedirs(save_dir, exist_ok=True)
    dummy_model = torch.zeros(0)
    device = Evaler.reload_device(device, dummy_model, task)
    data = Evaler.reload_dataset(data) if isinstance(data, str) else data
    val = Evaler(data, batch_size, img_size, None, None, device, False,
        save_dir)
    dataloader, pred_result = val.eval_trt(weights)
    eval_result = val.eval_model(pred_result, dummy_model, dataloader, task)
    return eval_result
