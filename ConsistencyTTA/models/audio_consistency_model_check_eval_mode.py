def check_eval_mode(self):
    super().check_eval_mode()
    for model, name in zip([self.student_target_unet, self.vae], [
        'student_target_unet', 'vae']):
        assert model.training == False, f'The {name} is not in eval mode.'
        for param in model.parameters():
            assert param.requires_grad == False, f'The {name} is not frozen.'
