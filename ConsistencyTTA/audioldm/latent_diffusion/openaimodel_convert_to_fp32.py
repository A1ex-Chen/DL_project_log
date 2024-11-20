def convert_to_fp32(self):
    """
        Convert the torso of the model to float32.
        """
    self.input_blocks.apply(convert_module_to_f32)
    self.middle_block.apply(convert_module_to_f32)
