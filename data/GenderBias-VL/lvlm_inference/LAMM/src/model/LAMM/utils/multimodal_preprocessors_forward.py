def forward(self, imu):
    imu = imu.unfold(-1, self.kernel_size, self.kernel_size).permute(0, 2, 1, 3
        )
    imu = imu.reshape(imu.size(0), imu.size(1), -1)
    imu_tokens = self.tokenize_input_and_cls_pos(imu, self.imu_stem)
    return_dict = {'trunk': {'tokens': imu_tokens}, 'head': {}}
    return return_dict
