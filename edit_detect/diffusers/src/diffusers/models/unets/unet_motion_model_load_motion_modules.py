def load_motion_modules(self, motion_adapter: Optional[MotionAdapter]) ->None:
    for i, down_block in enumerate(motion_adapter.down_blocks):
        self.down_blocks[i].motion_modules.load_state_dict(down_block.
            motion_modules.state_dict())
    for i, up_block in enumerate(motion_adapter.up_blocks):
        self.up_blocks[i].motion_modules.load_state_dict(up_block.
            motion_modules.state_dict())
    if hasattr(self.mid_block, 'motion_modules'):
        self.mid_block.motion_modules.load_state_dict(motion_adapter.
            mid_block.motion_modules.state_dict())
