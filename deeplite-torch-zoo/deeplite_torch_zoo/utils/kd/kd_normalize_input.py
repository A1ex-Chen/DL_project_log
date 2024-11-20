def normalize_input(self, input, student_model):
    if hasattr(student_model, 'module'):
        model_s = student_model.module
    else:
        model_s = student_model
    input_kd = input
    if hasattr(model_s, 'default_cfg'):
        mean_student = model_s.default_cfg['mean']
        std_student = model_s.default_cfg['std']
        if (mean_student != self.mean_model_kd or std_student != self.
            std_model_kd):
            std = self.std_model_kd[0] / std_student[0], self.std_model_kd[1
                ] / std_student[1], self.std_model_kd[2] / std_student[2]
            transform_std = T.Normalize(mean=(0, 0, 0), std=std)
            mean = self.mean_model_kd[0] - mean_student[0], self.mean_model_kd[
                1] - mean_student[1], self.mean_model_kd[2] - mean_student[2]
            transform_mean = T.Normalize(mean=mean, std=(1, 1, 1))
            input_kd = transform_mean(transform_std(input))
    return input_kd
