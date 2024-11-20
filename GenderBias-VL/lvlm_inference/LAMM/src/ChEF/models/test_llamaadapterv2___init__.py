def __init__(self, model_path, device='cuda', max_seq_len=1024,
    max_batch_size=40, **kwargs) ->None:
    llama_dir = model_path
    model, preprocess = llama_adapter_v2.load('LORA-BIAS-7B', llama_dir,
        download_root=llama_dir, max_seq_len=max_seq_len, max_batch_size=
        max_batch_size, device='cpu')
    self.img_transform = preprocess
    self.model = model.eval()
    self.tokenizer = self.model.tokenizer
    self.move_to_device(device)
