def __init__(self, ckpt_id: str, fp16: bool, use_safetensors: bool):
    self.logger = logging.get_logger('diffusers-cli/fp16_safetensors')
    self.ckpt_id = ckpt_id
    self.local_ckpt_dir = f'/tmp/{ckpt_id}'
    self.fp16 = fp16
    self.use_safetensors = use_safetensors
    if not self.use_safetensors and not self.fp16:
        raise NotImplementedError(
            'When `use_safetensors` and `fp16` both are False, then this command is of no use.'
            )
