def enable_npu_flash_attention(self) ->None:
    """
        Enable npu flash attention from torch_npu

        """
    self.set_use_npu_flash_attention(True)
