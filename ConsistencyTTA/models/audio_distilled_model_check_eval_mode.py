def check_eval_mode(self):
    models = [self.text_encoder, self.teacher_unet, self.student_ema_unet]
    names = ['text_encoder', 'teacher_unet', 'student_ema_unet']
    for model, name in zip(models if self.freeze_text_encoder else models[1
        :], names if self.freeze_text_encoder else names[1:]):
        assert model.training == False, f'The {name} is not in eval mode.'
        for param in model.parameters():
            assert param.requires_grad == False, f'The {name} is not frozen.'
