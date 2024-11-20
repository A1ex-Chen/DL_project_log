def convert_to_fp16(self):
    """
        Convert the torso of the model to float16.
        """
    self.input_blocks.apply(convert_module_to_f16)
    self.middle_block.apply(convert_module_to_f16)
