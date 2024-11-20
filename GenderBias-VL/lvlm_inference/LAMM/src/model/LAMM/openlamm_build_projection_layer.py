def build_projection_layer(self):
    super().build_projection_layer()
    if self.stage == 2:
        print('Load projector weights for stage 2 training')
        self.load_stage1_weights(self.args['llm_proj_path'])
