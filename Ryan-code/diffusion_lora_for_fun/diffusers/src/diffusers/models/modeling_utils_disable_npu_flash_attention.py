def disable_npu_flash_attention(self) ->None:
    """
        disable npu flash attention from torch_npu

        """
    self.set_use_npu_flash_attention(False)
