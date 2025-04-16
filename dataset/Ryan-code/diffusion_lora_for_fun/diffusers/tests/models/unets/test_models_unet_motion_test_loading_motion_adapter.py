def test_loading_motion_adapter(self):
    model = self.model_class()
    adapter = MotionAdapter()
    model.load_motion_modules(adapter)
    for idx, down_block in enumerate(model.down_blocks):
        adapter_state_dict = adapter.down_blocks[idx
            ].motion_modules.state_dict()
        for param_name, param_value in down_block.motion_modules.named_parameters(
            ):
            self.assertTrue(torch.equal(adapter_state_dict[param_name],
                param_value))
    for idx, up_block in enumerate(model.up_blocks):
        adapter_state_dict = adapter.up_blocks[idx].motion_modules.state_dict()
        for param_name, param_value in up_block.motion_modules.named_parameters(
            ):
            self.assertTrue(torch.equal(adapter_state_dict[param_name],
                param_value))
    mid_block_adapter_state_dict = adapter.mid_block.motion_modules.state_dict(
        )
    for param_name, param_value in model.mid_block.motion_modules.named_parameters(
        ):
        self.assertTrue(torch.equal(mid_block_adapter_state_dict[param_name
            ], param_value))
